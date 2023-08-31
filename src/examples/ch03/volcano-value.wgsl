const pi:f32 = 3.1415926;

struct ValueBuffer{
    values: array<f32>,
}
@group(0) @binding(0) var<storage, read_write> valueBuffer : ValueBuffer;

struct IntParams {
    resolution: u32,
    octaves: u32,
    octaves1: u32,
    seed: u32,
}
@group(0) @binding(1) var<uniform> ips : IntParams;

struct FloatParams {     
    offset: vec3f, 
    noiseScale: f32,
    noiseWeight: f32,    
    weightMultiplier: f32,
    floorOffset: f32,
    noiseScale1: f32,
    noiseWeight1: f32,    
    weightMultiplier1: f32,
    floorOffset1: f32,
}
@group(0) @binding(2) var<uniform> fps : FloatParams;

const terrainSize: f32 = 40.0;
const lacunarity: f32 = 2.0;
const persistence: f32 = 0.5;

/*fn positionAt(index : vec3u) -> vec3f {
    let vstep = vec3(terrainSize)/(f32(ips.resolution) - 1.0);
    return vec3(-0.5 * terrainSize) + (vstep * vec3<f32>(index.xyz));
}*/

var<private> vmin: vec3f;
var<private> vmax: vec3f;

fn positionAt(index : vec3u) -> vec3f {
    vmin = vec3(-0.5 * terrainSize);
    vmax = vec3(0.5 * terrainSize);
    let vstep = (vmax-vmin)/(f32(ips.resolution) - 1.0);
    return vmin + (vstep * vec3<f32>(index.xyz));
}

fn getIdx(id: vec3u) -> u32 {
    return id.x + ips.resolution * ( id.y + id.z * ips.resolution);
}

fn terrainFunc(position: vec3f) -> f32 {  
    var noise = 0.0;
    var frequency = fps.noiseScale1/100.0;
    var amplitude = 1.0;
    var weight = 1.0;

    for(var i = 0u; i < ips.octaves1; i = i + 1u){
        let rand = rand3d(f32(i + ips.seed));        
        let offset = 1000.0 * rand;
        let n = simplex3d((position + offset)*frequency );
        var v = 1.0 - abs(n);
        v = v * v;
        v *= weight;
        weight = max(min(v * fps.weightMultiplier1, 1.0), 0.0);
        noise += v * amplitude;
        amplitude *= persistence;
        frequency *= lacunarity;
    }
    return -(position.y + fps.floorOffset1) + noise * fps.noiseWeight1;
}

fn volcanoFunc(position: vec3f) -> f32 {
    var noise = 0.0;
    var frequency = fps.noiseScale/100.0;
    var amplitude = 1.0;
    var weight = 1.0;

    let r = sqrt(position.x * position.x + position.z * position.z);
    var offset1 = vec3(0.0);
    if(r < 9.0 + 1.5 * sin(0.05*fps.offset.y)) { offset1 = fps.offset; }  
    for(var i = 0u; i < ips.octaves; i = i + 1u){
        let rand = rand3d(f32(i + ips.seed));     
        let offset = 1000.0 * rand;
        let n = simplex3d((position  + offset1 + offset)*frequency );
        var v = 1.0 - abs(n);
        v = v * v;
        v *= weight;
        weight = max(min(v * fps.weightMultiplier, 1.0), 0.0);
        noise += v * amplitude;
        amplitude *= persistence;
        frequency *= lacunarity;
    }
    return -(position.y + fps.floorOffset) + noise * fps.noiseWeight;
}

fn combinedFunc(position: vec3f) -> f32 {
    let terrain = terrainFunc(position);
    let volcano = volcanoFunc(position);
    let r = sqrt(position.x * position.x + position.z * position.z);
    var val: f32;
    if(r < 10.0 + 0.5*sin(0.02*fps.offset.y)) {
        val = volcano;
    } else {
        val = terrain;
    }
    return val;
}

@compute @workgroup_size(8, 8, 8)
fn cs_main(@builtin(global_invocation_id) id : vec3u) {
    var position = positionAt(id);
    var y = (position.y - vmin.y)/(vmax.y - vmin.y);
    if(y <= 0.0) {
        position.y = 0.0;
    }
    let idx = getIdx(id);
    valueBuffer.values[idx] = combinedFunc(position);
}