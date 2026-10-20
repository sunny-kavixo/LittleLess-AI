#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdio>

static id<MTLDevice> g_device = nil;
static id<MTLCommandQueue> g_queue = nil;
static id<MTLComputePipelineState> g_pipeline = nil;

static const char* METAL_SOURCE = R"(
#include <metal_stdlib>
using namespace metal;

kernel void packed_dual_basis_gemv(
    device const uchar* p1            [[buffer(0)]],
    device const uchar* p2            [[buffer(1)]],
    device const half* s1             [[buffer(2)]],
    device const half* s2             [[buffer(3)]],
    device const half* x              [[buffer(4)]],
    device half* y                    [[buffer(5)]],
    constant uint& M                  [[buffer(6)]],
    constant uint& N                  [[buffer(7)]],
    uint id                           [[thread_position_in_grid]]
) {
    if (id >= M) return;
    
    uint bytes_per_row = N / 4;
    device const uchar* row_p1 = p1 + id * bytes_per_row;
    device const uchar* row_p2 = p2 + id * bytes_per_row;
    
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    
    for (uint jb = 0; jb < bytes_per_row; ++jb) {
        uint j = jb * 4;
        uchar b1 = row_p1[jb];
        uchar b2 = row_p2[jb];
        
        int c0 = (((b1 >> 0) & 1) << 1) | ((b1 >> 1) & 1);
        int c1 = (((b1 >> 2) & 1) << 1) | ((b1 >> 3) & 1);
        int c2 = (((b1 >> 4) & 1) << 1) | ((b1 >> 5) & 1);
        int c3 = (((b1 >> 6) & 1) << 1) | ((b1 >> 7) & 1);
        
        sum1 += (float)(c0 - 1) * (float)x[j + 0] +
                (float)(c1 - 1) * (float)x[j + 1] +
                (float)(c2 - 1) * (float)x[j + 2] +
                (float)(c3 - 1) * (float)x[j + 3];
                
        int d0 = (((b2 >> 0) & 1) << 1) | ((b2 >> 1) & 1);
        int d1 = (((b2 >> 2) & 1) << 1) | ((b2 >> 3) & 1);
        int d2 = (((b2 >> 4) & 1) << 1) | ((b2 >> 5) & 1);
        int d3 = (((b2 >> 6) & 1) << 1) | ((b2 >> 7) & 1);
        
        sum2 += (float)(d0 - 1) * (float)x[j + 0] +
                (float)(d1 - 1) * (float)x[j + 1] +
                (float)(d2 - 1) * (float)x[j + 2] +
                (float)(d3 - 1) * (float)x[j + 3];
    }
    
    y[id] = (half)((float)s1[id] * sum1 + (float)s2[id] * sum2);
}
)";

static void init_metal_if_needed() {
    if (g_pipeline != nil) return;
    
    @autoreleasepool {
        g_device = MTLCreateSystemDefaultDevice();
        g_queue = [g_device newCommandQueue];
        
        NSError* error = nil;
        NSString* src = [NSString stringWithUTF8String:METAL_SOURCE];
        id<MTLLibrary> lib = [g_device newLibraryWithSource:src options:nil error:&error];
        if (!lib) {
            NSLog(@"Metal compilation error: %@", error);
            return;
        }
        
        id<MTLFunction> func = [lib newFunctionWithName:@"packed_dual_basis_gemv"];
        g_pipeline = [g_device newComputePipelineStateWithFunction:func error:&error];
        if (!g_pipeline) {
            NSLog(@"Metal pipeline error: %@", error);
        }
    }
}

extern "C" {

/**
 * Executes direct packed dual-basis GEMV on Apple Silicon GPU using Metal!
 * Zero dense weight matrices are allocated or reconstructed.
 */
void metal_packed_gemv(
    const uint8_t* p1,
    const uint8_t* p2,
    const uint16_t* s1, // half
    const uint16_t* s2, // half
    const uint16_t* x,  // half
    uint16_t* y,        // half
    int M,
    int N
) {
    init_metal_if_needed();
    
    @autoreleasepool {
        uint32_t uM = (uint32_t)M;
        uint32_t uN = (uint32_t)N;
        uint32_t bytes_per_row = uN / 4;
        uint32_t packed_size = uM * bytes_per_row;
        
        // Zero-copy Shared Metal Buffers wrapping the raw memory pointers
        id<MTLBuffer> buf_p1 = [g_device newBufferWithBytesNoCopy:(void*)p1 length:packed_size options:MTLResourceStorageModeShared deallocator:nil];
        id<MTLBuffer> buf_p2 = [g_device newBufferWithBytesNoCopy:(void*)p2 length:packed_size options:MTLResourceStorageModeShared deallocator:nil];
        id<MTLBuffer> buf_s1 = [g_device newBufferWithBytesNoCopy:(void*)s1 length:uM * sizeof(uint16_t) options:MTLResourceStorageModeShared deallocator:nil];
        id<MTLBuffer> buf_s2 = [g_device newBufferWithBytesNoCopy:(void*)s2 length:uM * sizeof(uint16_t) options:MTLResourceStorageModeShared deallocator:nil];
        id<MTLBuffer> buf_x  = [g_device newBufferWithBytesNoCopy:(void*)x length:uN * sizeof(uint16_t) options:MTLResourceStorageModeShared deallocator:nil];
        id<MTLBuffer> buf_y  = [g_device newBufferWithBytesNoCopy:(void*)y length:uM * sizeof(uint16_t) options:MTLResourceStorageModeShared deallocator:nil];
        
        id<MTLCommandBuffer> cmdBuf = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
        
        [encoder setComputePipelineState:g_pipeline];
        [encoder setBuffer:buf_p1 offset:0 atIndex:0];
        [encoder setBuffer:buf_p2 offset:0 atIndex:1];
        [encoder setBuffer:buf_s1 offset:0 atIndex:2];
        [encoder setBuffer:buf_s2 offset:0 atIndex:3];
        [encoder setBuffer:buf_x  offset:0 atIndex:4];
        [encoder setBuffer:buf_y  offset:0 atIndex:5];
        [encoder setBytes:&uM length:sizeof(uint32_t) atIndex:6];
        [encoder setBytes:&uN length:sizeof(uint32_t) atIndex:7];
        
        NSUInteger threadGroupSize = 64;
        if (threadGroupSize > g_pipeline.maxTotalThreadsPerThreadgroup) {
            threadGroupSize = g_pipeline.maxTotalThreadsPerThreadgroup;
        }
        MTLSize threadgroups = MTLSizeMake((uM + threadGroupSize - 1) / threadGroupSize, 1, 1);
        MTLSize threadsPerGroup = MTLSizeMake(threadGroupSize, 1, 1);
        
        [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerGroup];
        [encoder endEncoding];
        
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];
    }
}

}
