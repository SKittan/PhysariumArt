use bytemuck::{Pod, Zeroable};
use nannou::prelude::*;

// The vertex type that we will use to represent a point on our triangle.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Vertex {
    pub position: [f32; 2],
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct Uniforms {  // parameter
    pub n_agents: usize,
    pub size_x: u32,
    pub size_y: u32,
    pub decay: f32
}

pub struct Physarum {
    pub agents: wgpu::Buffer,
    pub slime_agents: wgpu::Buffer,
    pub slime_slime: wgpu::Buffer,
    pub slime_size: wgpu::BufferAddress,
    pub uniform_buffer: wgpu::Buffer,
    pub bind_group_physarum: wgpu::BindGroup,
    pub bind_group_slime: wgpu::BindGroup,
    pub compute_physarum: wgpu::ComputePipeline,
    pub compute_slime: wgpu::ComputePipeline,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct Agent {
    pub x: f32,
    pub y: f32,
    pub phi: f32,
    pub sens: u32  // sensor values: 0, 1, 2, 4, 1+2, 2+4
}
unsafe impl Zeroable for Agent {}
unsafe impl Pod for Agent {}

pub fn create_bind_group_layout_compute(device: &wgpu::Device,
                                        read_only_first: bool)
-> wgpu::BindGroupLayout
{
    let storage_dynamic = false;
    let storage_readonly = false;
    let uniform_dynamic = false;
    wgpu::BindGroupLayoutBuilder::new()
        .storage_buffer(
            wgpu::ShaderStages::COMPUTE,
            storage_dynamic,
            read_only_first,
        )
        .storage_buffer(
            wgpu::ShaderStages::COMPUTE,
            storage_dynamic,
            storage_readonly,
        )
        .uniform_buffer(wgpu::ShaderStages::COMPUTE, uniform_dynamic)
        .build(device)
}

pub fn create_bind_group_layout_render(
    device: &wgpu::Device)
-> wgpu::BindGroupLayout
{
    let storage_dynamic = false;
    let storage_readonly = true;
    let uniform_dynamic = false;
    wgpu::BindGroupLayoutBuilder::new()
        .storage_buffer(
            wgpu::ShaderStages::FRAGMENT,
            storage_dynamic,
            storage_readonly,
        )
        .uniform_buffer(wgpu::ShaderStages::FRAGMENT, uniform_dynamic)
        .build(device)
}

pub fn create_physarum_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    agents: &wgpu::Buffer,
    agent_size: &usize,
    slime: &wgpu::Buffer,
    slime_size: &usize,
    uniform_buffer: &wgpu::Buffer)
-> wgpu::BindGroup
{
    wgpu::BindGroupBuilder::new()
        .buffer::<Agent>(agents, 0..*agent_size)
        .buffer::<f32>(slime,  0..*slime_size)
        .buffer::<Uniforms>(uniform_buffer, 0..1)
        .build(device, layout)
}

pub fn create_slime_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    slime_in: &wgpu::Buffer,
    slime_out: &wgpu::Buffer,  // intermediate result for dissipation
    slime_size: &usize,
    uniform_buffer: &wgpu::Buffer)
-> wgpu::BindGroup
{
    wgpu::BindGroupBuilder::new()
        .buffer::<f32>(slime_in, 0..*slime_size)
        .buffer::<f32>(slime_out, 0..*slime_size)
        .buffer::<Uniforms>(uniform_buffer, 0..1)
        .build(device, layout)
}

pub fn create_render_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    buffer: &wgpu::Buffer,
    buffer_size: wgpu::BufferAddress,
    uniform_buffer: &wgpu::Buffer)
-> wgpu::BindGroup
{
    let buffer_size_bytes = std::num::NonZeroU64::new(buffer_size).unwrap();
    wgpu::BindGroupBuilder::new()
        .buffer_bytes(buffer, 0, Some(buffer_size_bytes))
        .buffer::<Uniforms>(uniform_buffer, 0..1)
        .build(device, layout)
}

pub fn create_pipeline_layout(
    device: &wgpu::Device,
    bind_group_layout: &wgpu::BindGroupLayout,
    label: &str)
-> wgpu::PipelineLayout
{
    let desc = wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    };
    device.create_pipeline_layout(&desc)
}

pub fn create_compute_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    cs_mod: &wgpu::ShaderModule,
    label: &str
) -> wgpu::ComputePipeline
{
    let desc = wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(layout),
        module: &cs_mod,
        entry_point: "main",
    };
    device.create_compute_pipeline(&desc)
}

pub fn create_render_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    vs_mod: &wgpu::ShaderModule,
    fs_mod: &wgpu::ShaderModule,
    dst_format: wgpu::TextureFormat,
    sample_count: u32)
-> wgpu::RenderPipeline
{
    wgpu::RenderPipelineBuilder::from_layout(layout, vs_mod)
        .fragment_shader(fs_mod)
        .color_format(dst_format)
        .add_vertex_buffer::<Vertex>(&wgpu::vertex_attr_array![0 => Float32x2])
        .sample_count(sample_count)
        .primitive_topology(wgpu::PrimitiveTopology::TriangleStrip)
        .build(device)
}