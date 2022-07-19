use bytemuck::{Pod, Zeroable};
use wgpu;


// The vertex type that we will use to represent a point on our triangle.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Vertex {
    pub position: [f32; 2],
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct Uniforms {  // parameter
    pub n_agents: u32,
    pub size_x: u32,
    pub size_y: u32,
    pub deposit: f32,
    pub decay: f32,
    pub v: f32,
    pub d_phi_sens: f32,  // Stepping of sensor angle
    pub phi_sens_0: f32,  // Start of sensor angle
    pub phi_sens_1: f32,  // End of sensor angle
    pub sens_range_min: f32,
    pub sens_range_max: f32,
    pub sense_steps: f32,
    pub seed_1: f32,
    pub seed_2: f32
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct Agent {
    pub x: f32,
    pub y: f32,
    pub phi: f32
}

unsafe impl Zeroable for Agent {}
unsafe impl Pod for Agent {}
unsafe impl Zeroable for Uniforms {}
unsafe impl Pod for Uniforms {}
unsafe impl Zeroable for Vertex {}
unsafe impl Pod for Vertex {}


pub fn create_bind_group_layout_compute_agents(device: &wgpu::Device)
-> wgpu::BindGroupLayout
{
    device.create_bind_group_layout(
        &wgpu::BindGroupLayoutDescriptor{
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage {
                            read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None },
                    count: None
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage {
                            read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None },
                    count: None
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage {
                            read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None },
                    count: None
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None },
                    count: None
                }
            ],
            label: None,
        }
    )
}

pub fn create_bind_group_layout_compute_slime(device: &wgpu::Device)
-> wgpu::BindGroupLayout
{
    device.create_bind_group_layout(
        &wgpu::BindGroupLayoutDescriptor{
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage {
                            read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None },
                    count: None
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage {
                            read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None },
                    count: None
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None },
                    count: None
                }
            ],
            label: None,
        }
    )
}

pub fn create_bind_group_layout_render(
    device: &wgpu::Device)
-> wgpu::BindGroupLayout
{
    device.create_bind_group_layout(
        &wgpu::BindGroupLayoutDescriptor{
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage {
                            read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None },
                    count: None
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None },
                    count: None
                }
            ],
            label: Some("Render Layout"),
        }
    )
}

pub fn create_physarum_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    agents: &wgpu::Buffer,
    slime_in: &wgpu::Buffer,
    slime_out: &wgpu::Buffer,
    uniform_buffer: &wgpu::Buffer)
-> wgpu::BindGroup
{
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout,
        label: Some("Physarum BG"),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: agents.as_entire_binding()
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: slime_in.as_entire_binding()
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: slime_out.as_entire_binding()
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: uniform_buffer.as_entire_binding()
            }
        ]
    })
}

pub fn create_slime_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    slime_in: &wgpu::Buffer,
    slime_out: &wgpu::Buffer,  // intermediate result for dissipation
    uniform_buffer: &wgpu::Buffer)
-> wgpu::BindGroup
{
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout,
        label: Some("Slime BG"),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: slime_in.as_entire_binding()
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: slime_out.as_entire_binding()
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: uniform_buffer.as_entire_binding()
            }
        ]
    })
}

pub fn create_render_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    slime: &wgpu::Buffer,
    uniform_buffer: &wgpu::Buffer)
-> wgpu::BindGroup
{
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout,
        label: Some("Render BG"),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: slime.as_entire_binding()
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: uniform_buffer.as_entire_binding()
            }
        ]
    })
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

impl Vertex {
    pub fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x2,
                }
            ]
        }
    }
}
