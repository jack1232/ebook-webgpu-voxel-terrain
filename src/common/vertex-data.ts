export const getCubeData = (side = 2) => {
    let s2 = side / 2;
    let positions = new Float32Array([
        s2,  s2,  s2,   // index 0
        s2,  s2, -s2,   // index 1
        s2, -s2,  s2,   // index 2
        s2, -s2, -s2,   // index 3
       -s2,  s2, -s2,   // index 4
       -s2,  s2,  s2,   // index 5
       -s2, -s2, -s2,   // index 6
       -s2, -s2,  s2,   // index 7
       -s2,  s2, -s2,   // index 8
        s2,  s2, -s2,   // index 9
       -s2,  s2,  s2,   // index 10
        s2,  s2,  s2,   // index 11
       -s2, -s2,  s2,   // index 12
        s2, -s2,  s2,   // index 13
       -s2, -s2, -s2,   // index 14
        s2, -s2, -s2,   // index 15
       -s2,  s2,  s2,   // index 16
        s2,  s2,  s2,   // index 17
       -s2, -s2,  s2,   // index 18
        s2, -s2,  s2,   // index 19
        s2,  s2, -s2,   // index 20
       -s2,  s2, -s2,   // index 21
        s2, -s2, -s2,   // index 22
       -s2, -s2, -s2,   // index 23
    ]); 

    let colors = new Float32Array([
        1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0,
        0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1,
        0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1,
        0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0,
        0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1,
        1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0
    ]);

    let normals = new Float32Array([
        1,  0,  0,  1,  0,  0,  1,  0,  0,  1,  0,  0,
       -1,  0,  0, -1,  0,  0, -1,  0,  0, -1,  0,  0,
        0,  1,  0,  0,  1,  0,  0,  1,  0,  0,  1,  0,
        0, -1,  0,  0, -1,  0,  0, -1,  0,  0, -1,  0,
        0,  0,  1,  0,  0,  1,  0,  0,  1,  0,  0,  1,
        0,  0, -1,  0,  0, -1,  0,  0, -1,  0,  0, -1,
    ]);

    let indices = new Uint32Array([     // triangle indices
         0,  2,  1,
         2,  3,  1,
         4,  6,  5,
         6,  7,  5,
         8, 10,  9,
        10, 11,  9,
        12, 14, 13,
        14, 15, 13,
        16, 18, 17,
        18, 19, 17,
        20, 22, 21,
        22, 23, 21,
    ]);
 
    return {positions, colors, normals, indices};
}