import vsShader from '../../common/shader-vert.wgsl';
import fsShader from '../../common/shader-frag.wgsl';
import csVolcanoShader from './volcano.wgsl';
import csValueShader from './volcano-value.wgsl';
import csNoiseShader from '../../common/noise3d.wgsl';
import { EdgeTable, TriTable } from '../../common/marching-cubes-table'; 
import * as ws from 'webgpu-simplified';
import { vec3 } from 'gl-matrix';

let resolution = 152;
let marchingCubeCells = (resolution - 1) * (resolution - 1) * (resolution - 1);
let vertexCount = 3 * 12 * marchingCubeCells;
let vertexBufferSize = Float32Array.BYTES_PER_ELEMENT * vertexCount;
let indexCount = 15 * marchingCubeCells;
let indexBufferSize = Uint32Array.BYTES_PER_ELEMENT * indexCount;
let indirectArray:Uint32Array;

const createPipeline = async (init:ws.IWebGPUInit): Promise<ws.IPipeline> => {
    const descriptor = ws.createRenderPipelineDescriptor({
        init, vsShader, fsShader,
        buffers: ws.setVertexBuffers(['float32x3', 'float32x3', 'float32x3']),//pos, norm, col 
    })
    const pipeline = await init.device.createRenderPipelineAsync(descriptor);

    // uniform buffer for transform matrix
    const  vertUniformBuffer = ws.createBuffer(init.device, 192);

    // uniform buffer for light 
    const lightUniformBuffer = ws.createBuffer(init.device, 48);

    // uniform buffer for material
    const materialUniformBuffer = ws.createBuffer(init.device, 16);
    
    // uniform bind group for vertex shader
    const vertBindGroup = ws.createBindGroup(init.device, pipeline.getBindGroupLayout(0), [vertUniformBuffer]);
    
    // uniform bind group for fragment shader
    const fragBindGroup = ws.createBindGroup(init.device, pipeline.getBindGroupLayout(1), 
        [lightUniformBuffer, materialUniformBuffer]);

    // create depth view
    const depthTexture = ws.createDepthTexture(init);

    // create texture view for MASS (count = 4)
    const msaaTexture = ws.createMultiSampleTexture(init);

    return {
        pipelines: [pipeline],
        uniformBuffers: [
            vertUniformBuffer,    // for vertex
            lightUniformBuffer,   // for fragmnet
            materialUniformBuffer      
        ],
        uniformBindGroups: [vertBindGroup, fragBindGroup],
        depthTextures: [depthTexture],
        gpuTextures: [msaaTexture],
    };
}

const createComputeValuePipeline = async (device: GPUDevice): Promise<ws.IPipeline> => {    
    const valueShader = csNoiseShader.concat(csValueShader);
    const descriptor = ws.createComputePipelineDescriptor(device, valueShader);
    const csPipeline = await device.createComputePipelineAsync(descriptor);

    const volumeElements = resolution * resolution * resolution;
    const valueBufferSize = Float32Array.BYTES_PER_ELEMENT * volumeElements;
    const valueBuffer = ws.createBuffer(device, valueBufferSize, ws.BufferType.Storage);

    const intParamsBufferSize = 
        1 * 4 + // resolution: u32
        1 * 4 + // octaves: u32
        1 * 4 + // octaves1: u32
        1 * 4 + // seed: u32
        0;      
    const intBuffer = ws.createBuffer(device, intParamsBufferSize);
    
    const floatParamsBufferSize = 
        3 * 4 + // offset: vec3<f32>
        1 + 4 + // noiseScale: f32
        1 + 4 + // noiseWeight: f32
        1 + 4 + // weightMultiplier: f32
        1 + 4 + // floorOffset: f32
        1 + 4 + // noiseScale1: f32
        1 + 4 + // noiseWeight1: f32
        1 + 4 + // weightMultiplier1: f32
        1 + 4 + // floorOffset1: f32
        1 * 4 + // padding
        0;      
    const floatBuffer = ws.createBuffer(device, floatParamsBufferSize);
   
    const csBindGroup = ws.createBindGroup(device, csPipeline.getBindGroupLayout(0), 
        [valueBuffer, intBuffer, floatBuffer]);
    
    return {
        csPipelines: [csPipeline],
        vertexBuffers: [valueBuffer],
        uniformBuffers: [intBuffer, floatBuffer],
        uniformBindGroups: [csBindGroup],        
    };
}

const createComputePipeline = async (device: GPUDevice, valueBuffer: GPUBuffer): Promise<ws.IPipeline> => {     
    const descriptor = ws.createComputePipelineDescriptor(device, csVolcanoShader);
    const csPipeline = await device.createComputePipelineAsync(descriptor);
    
    const tableBuffer = device.createBuffer({
        size: (EdgeTable.length + TriTable.length)*Int32Array.BYTES_PER_ELEMENT,
        usage: GPUBufferUsage.STORAGE,
        mappedAtCreation: true,
    });
    const tableArray = new Int32Array(tableBuffer.getMappedRange());
    tableArray.set(EdgeTable);
    tableArray.set(TriTable, EdgeTable.length);
    tableBuffer.unmap();

    const positionBuffer = ws.createBuffer(device, vertexBufferSize, ws.BufferType.VertexStorage);
    const normalBuffer = ws.createBuffer(device, vertexBufferSize, ws.BufferType.VertexStorage);
    const colorBuffer = ws.createBuffer(device, vertexBufferSize, ws.BufferType.VertexStorage);
    const indexBuffer = ws.createBuffer(device, indexBufferSize, ws.BufferType.IndexStorage);
   
    indirectArray = new Uint32Array(4);
    indirectArray[0] = 500;
    const indirectBuffer = ws.createBuffer(device, indirectArray.byteLength, ws.BufferType.IndirectStorage);
    
    const intParamsBufferSize = 
        1 * 4 + // resolution: u32
        3 * 4 + // padding
        0;      
    const intBuffer = ws.createBuffer(device, intParamsBufferSize);
   
    const floatParamsBufferSize = 
        1 * 4 + // isolevel: f32
        3 * 4 + // padding
        0;      
    const floatBuffer = ws.createBuffer(device, floatParamsBufferSize);

    const csBindGroup = ws.createBindGroup(device, csPipeline.getBindGroupLayout(0), [
        tableBuffer, valueBuffer, positionBuffer, normalBuffer, colorBuffer, indexBuffer, 
        indirectBuffer, intBuffer, floatBuffer
    ]);

    return {
        csPipelines: [csPipeline],
        vertexBuffers: [positionBuffer, normalBuffer, colorBuffer, indexBuffer],
        uniformBuffers: [intBuffer, floatBuffer, indirectBuffer],
        uniformBindGroups: [csBindGroup],        
    };
}

const draw = async (init:ws.IWebGPUInit, p:ws.IPipeline, p2:ws.IPipeline, p3: ws.IPipeline) => {  
    const commandEncoder =  init.device.createCommandEncoder();
    const wsize = 8; // set workgroup size

    // compute pass
    {
        const csPass = commandEncoder.beginComputePass();
        csPass.setPipeline(p2.csPipelines[0]);
        csPass.setBindGroup(0, p2.uniformBindGroups[0]);
        csPass.dispatchWorkgroups(Math.ceil(resolution / wsize), Math.ceil(resolution / wsize), 
            Math.ceil(resolution / wsize));

        csPass.setPipeline(p3.csPipelines[0]);
        csPass.setBindGroup(0, p3.uniformBindGroups[0]);
        csPass.dispatchWorkgroups(Math.ceil(resolution / wsize), Math.ceil(resolution / wsize), 
            Math.ceil(resolution / wsize));
        csPass.end();
    }
    
    // render pass
    {
        const descriptor = ws.createRenderPassDescriptor({
            init,
            depthView: p.depthTextures[0].createView(),
            textureView: p.gpuTextures[0].createView(),
        });
        const renderPass = commandEncoder.beginRenderPass(descriptor);

        // draw surface
        renderPass.setPipeline(p.pipelines[0]);
        renderPass.setBindGroup(0, p.uniformBindGroups[0]);
        renderPass.setBindGroup(1, p.uniformBindGroups[1]);
        renderPass.setVertexBuffer(0, p3.vertexBuffers[0]);
        renderPass.setVertexBuffer(1, p3.vertexBuffers[1]);     
        renderPass.setVertexBuffer(2, p3.vertexBuffers[2]);     
        renderPass.setIndexBuffer(p3.vertexBuffers[3], 'uint32');
        renderPass.drawIndexed(indexCount);

        renderPass.end();
    }

    init.device.queue.submit([commandEncoder.finish()]);
}

const run = async () => {
    const canvas = document.getElementById('canvas-webgpu') as HTMLCanvasElement;
    const deviceDescriptor: GPUDeviceDescriptor = {
        requiredLimits:{
            maxStorageBufferBindingSize: 1024*1024*1024, //1024MB, defaulting to 128MB
            maxBufferSize: 1024*1024*1024, // 1024MB, defaulting to 256MB
            maxComputeInvocationsPerWorkgroup: 512 // defaulting to 256
        }
    }
    const init = await ws.initWebGPU({canvas, msaaCount: 4}, deviceDescriptor);

    var gui = ws.getDatGui();
    const params = {
        animateSpeed: 1,
        rotateSpeed: 1,
        seed: 1232,
        isolevel: 3,
        octaves: 10,
        noiseScale: 10,
        noiseWeight: 9,
        weightMultiplier: 24,
        floorOffset: 10,
        octaves1: 10,
        noiseScale1: 5,
        noiseWeight1: 5,
        weightMultiplier1: 3,
        floorOffset1: 10,       

        specularColor: '#aaaaaa',
        ambient: 0.5,
        diffuse: 0.7,
        specular: 0.4,
        shininess: 30,
    };
    
    let dataChanged = true;
        
    gui.add(params, 'animateSpeed', 0, 5, 0.1);    
    gui.add(params, 'rotateSpeed', 0, 5, 0.1);  
    
    var folder = gui.addFolder('Set Volcano Parameters');
    folder.open();
    folder.add(params, 'seed', 1, 65536, 1).onChange(()=>{dataChanged = true;}); 
    folder.add(params, 'isolevel', -20, 20, 0.1).onChange(()=>{dataChanged = true;}); 
    folder.add(params, 'octaves', 1, 20, 1).onChange(()=>{dataChanged = true;});
    folder.add(params, 'noiseScale', 1, 50, 1).onChange(()=>{dataChanged = true;});
    folder.add(params, 'noiseWeight', 0, 20, 0.2).onChange(()=>{dataChanged = true;});
    folder.add(params, 'weightMultiplier', 1, 10, 0.1).onChange(()=>{dataChanged = true;});
    folder.add(params, 'floorOffset', 1, 100, 0.2).onChange(()=>{dataChanged = true;});

    folder = gui.addFolder('Set Terrain Parameters');
    folder.open();
    folder.add(params, 'octaves1', 1, 20, 1).onChange(()=>{dataChanged = true;});
    folder.add(params, 'noiseScale1', 1, 50, 1).onChange(()=>{dataChanged = true;});
    folder.add(params, 'noiseWeight1', 0, 20, 0.2).onChange(()=>{dataChanged = true;});
    folder.add(params, 'weightMultiplier1', 1, 10, 0.1).onChange(()=>{dataChanged = true;});
    folder.add(params, 'floorOffset1', 1, 100, 0.2).onChange(()=>{dataChanged = true;});

    folder = gui.addFolder('Set Lighting Parameters');
    folder.open();
    folder.add(params, 'ambient', 0, 1, 0.02).onChange(()=>{dataChanged = true;});;  
    folder.add(params, 'diffuse', 0, 1, 0.02).onChange(()=>{dataChanged = true;});;  
    folder.addColor(params, 'specularColor').onChange(()=>{dataChanged = true;});;
    folder.add(params, 'specular', 0, 1, 0.02).onChange(()=>{dataChanged = true;});;  
    folder.add(params, 'shininess', 0, 300, 1).onChange(()=>{dataChanged = true;});;  

    const p = await createPipeline(init);
    let p2 = await createComputeValuePipeline(init.device);   
    let p3 = await createComputePipeline(init.device, p2.vertexBuffers[0]);
    
    let scale = 1;
    let rotation = [0,Math.PI/10,0] as vec3;
    let modelMat = ws.createModelMat([0,5,0], rotation, [scale,scale,scale]);
    let normalMat = ws.createNormalMat(modelMat);
    init.device.queue.writeBuffer(p.uniformBuffers[0], 64, modelMat as ArrayBuffer);  
    init.device.queue.writeBuffer(p.uniformBuffers[0], 128, normalMat as ArrayBuffer); 

    let vt = ws.createViewTransform([12, 12, 12]);
    let viewMat = vt.viewMat;
   
    let aspect = init.size.width / init.size.height;    
    let projectMat = ws.createProjectionMat(aspect);  
    let vpMat = ws.combineVpMat(viewMat, projectMat);

    var camera = ws.getCamera(canvas, vt.cameraOptions);
    let eyePosition = new Float32Array(vt.cameraOptions.eye);
    let lightDirection = new Float32Array([-0.5, -0.5, -0.5]);
    init.device.queue.writeBuffer(p.uniformBuffers[0], 0, vpMat as ArrayBuffer);

    // write light parameters to buffer 
    init.device.queue.writeBuffer(p.uniformBuffers[1], 0, lightDirection);
    init.device.queue.writeBuffer(p.uniformBuffers[1], 16, eyePosition);
   
    let start = performance.now();
    let stats = ws.getStats();

    const frame = async () => {     
        stats.begin();

        projectMat = ws.createProjectionMat(aspect); 
        if(camera.tick()){
            viewMat = camera.matrix;
            vpMat = ws.combineVpMat(viewMat, projectMat);
            eyePosition = new Float32Array(camera.eye.flat());
            init.device.queue.writeBuffer(p.uniformBuffers[0], 0, vpMat as ArrayBuffer);
            init.device.queue.writeBuffer(p.uniformBuffers[1], 16, eyePosition);
        }
        var dt = (performance.now() - start)/100;   
        rotation[1] = 0.02*dt * params.rotateSpeed; 
        modelMat = ws.createModelMat([0,5,0], rotation, [scale,scale,scale]);
        let normalMat = ws.createNormalMat(modelMat);
        init.device.queue.writeBuffer(p.uniformBuffers[0], 64, modelMat as ArrayBuffer);  
        init.device.queue.writeBuffer(p.uniformBuffers[0], 128, normalMat as ArrayBuffer); 
        
        if(dataChanged){
            // update uniform buffers for specular light color
            init.device.queue.writeBuffer(p.uniformBuffers[1], 32, ws.hex2rgb(params.specularColor));
            
                // update uniform buffer for material
            init.device.queue.writeBuffer(p.uniformBuffers[2], 0, new Float32Array([
                params.ambient, params.diffuse, params.specular, params.shininess
            ]));
            
            // update compute value pipeline
            init.device.queue.writeBuffer(p2.uniformBuffers[0], 0, new Uint32Array([
                resolution, 
                params.octaves, 
                params.octaves1,
                params.seed, 
            ]));

            // update complue pipeline
            init.device.queue.writeBuffer(p3.uniformBuffers[0], 0, new Uint32Array([
                resolution,
                0, // padding
                0, 
                0, 
            ]));

            init.device.queue.writeBuffer(p3.uniformBuffers[1], 0, new Float32Array([
                params.isolevel,
                0, // padding
                0,
                0,
            ]));          
            dataChanged = false;
        }

        init.device.queue.writeBuffer(p3.uniformBuffers[2], 0, indirectArray);

        init.device.queue.writeBuffer(p2.uniformBuffers[1], 0, new Float32Array([
            0, // offsets
            -dt*params.animateSpeed,
            0,
            params.noiseScale,
            params.noiseWeight,
            params.weightMultiplier,
            params.floorOffset,
            params.noiseScale1,
            params.noiseWeight1,
            params.weightMultiplier1,
            params.floorOffset1,
            0, //padding
        ]));

        draw(init, p, p2, p3);      

        requestAnimationFrame(frame);
        stats.end();
    };
    frame();
}

run();