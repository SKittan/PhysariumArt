use nannou::prelude::*;


// The vertex type that we will use to represent a point on our triangle.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Vertex {
    pub position: [f32; 2],
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct Uniforms {
    x_size: u32,
    y_size: f32,
}


pub fn create_bind_group_layout_buffer(device: &wgpu::Device)
-> wgpu::BindGroupLayout
{
    let storage_dynamic = false;
    let storage_readonly = false;
    let uniform_dynamic = false;
    wgpu::BindGroupLayoutBuilder::new()
        .storage_buffer(
            wgpu::ShaderStage::COMPUTE,
            storage_dynamic,
            storage_readonly,
        )
        .uniform_buffer(wgpu::ShaderStage::COMPUTE, uniform_dynamic)
        .build(device)
}

pub fn create_bind_group_layout_render(
    device: &wgpu::Device)
-> wgpu::BindGroupLayout {
    wgpu::BindGroupLayoutBuilder::new()
        .build(device)
}

pub fn create_bind_group_buffer(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    buffer: &wgpu::Buffer,
    buffer_size: wgpu::BufferAddress,
    uniform_buffer: &wgpu::Buffer)
-> wgpu::BindGroup {
    let buffer_size_bytes = std::num::NonZeroU64::new(buffer_size).unwrap();
    wgpu::BindGroupBuilder::new()
        .buffer_bytes(buffer, 0, Some(buffer_size_bytes))
        .buffer::<Uniforms>(uniform_buffer, 0..1)
        .build(device, layout)
}

pub fn create_bind_group_render(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout)
-> wgpu::BindGroup {
    wgpu::BindGroupBuilder::new().build(device, layout)
}

pub fn create_pipeline_layout(
    device: &wgpu::Device,
    bind_group_layout: &wgpu::BindGroupLayout)
-> wgpu::PipelineLayout {
    let desc = wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    };
    device.create_pipeline_layout(&desc)
}

pub fn create_render_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    vs_mod: &wgpu::ShaderModule,
    fs_mod: &wgpu::ShaderModule,
    dst_format: wgpu::TextureFormat,
    sample_count: u32)
-> wgpu::RenderPipeline {
    wgpu::RenderPipelineBuilder::from_layout(layout, vs_mod)
        .fragment_shader(fs_mod)
        .color_format(dst_format)
        .add_vertex_buffer::<Vertex>(&wgpu::vertex_attr_array![0 => Float2])
        .sample_count(sample_count)
        .primitive_topology(wgpu::PrimitiveTopology::TriangleStrip)
        .build(device)
}