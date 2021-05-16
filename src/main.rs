use nannou::prelude::*;

mod gpu_create;
use gpu_create::{create_bind_group_buffer, create_bind_group_layout_buffer,
                 create_bind_group_texture, create_bind_group_layout_texture,
                 create_render_pipeline, create_pipeline_layout,
                 Uniforms, Vertex};


struct Model {
    bind_group: wgpu::BindGroup,
    render_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
}

// The vertices that make up the rectangle to which the image will be drawn.
const VERTICES: [Vertex; 4] = [
    Vertex {
        position: [-1.0, 1.0],
    },
    Vertex {
        position: [-1.0, -1.0],
    },
    Vertex {
        position: [1.0, 1.0],
    },
    Vertex {
        position: [1.0, -1.0],
    },
];

fn main() {
    nannou::app(model).run();
}

fn model(app: &App) -> Model {
    let (size_x, size_y) = (512, 512);

    let w_id = app
        .new_window()
        .size(size_x, size_y)
        .view(view)
        .build()
        .unwrap();
    let window = app.window(w_id).unwrap();
    let device = window.swap_chain_device();
    let format = Frame::TEXTURE_FORMAT;
    let msaa_samples = window.msaa_samples();

    let vs_mod =
        wgpu::shader_from_spirv_bytes
            (device, include_bytes!("../Shader/passThrough.vert.spv"));
    let fs_mod =
        wgpu::shader_from_spirv_bytes(
            device, include_bytes!("../Shader/render.frag.spv"));

    let texture = wgpu::TextureBuilder::new()
                  .size([size_x, size_y])
                  .usage(wgpu::TextureUsage::RENDER_ATTACHMENT |
                         wgpu::TextureUsage::SAMPLED)
                  .sample_count(msaa_samples)
                  .format(wgpu::TextureFormat::Rgba16Float)
                  .build(device);
    let texture_view = texture.view().build();

    // Create the sampler for sampling from the source texture.
    let sampler_desc = wgpu::SamplerBuilder::new().into_descriptor();
    let sampler_filtering = wgpu::sampler_filtering(&sampler_desc);
    let sampler = device.create_sampler(&sampler_desc);

    let bind_group_layout =
        create_bind_group_layout_texture(device, texture_view.sample_type(),
                                         sampler_filtering);
    let bind_group = create_bind_group_texture(device, &bind_group_layout,
                                               &texture_view, &sampler);
    let pipeline_layout = create_pipeline_layout(device, &bind_group_layout);
    let render_pipeline = create_render_pipeline(
        device,
        &pipeline_layout,
        &vs_mod,
        &fs_mod,
        format,
        msaa_samples,
    );

    let vertices_bytes = vertices_as_bytes(&VERTICES[..]);
    let usage = wgpu::BufferUsage::VERTEX;
    let vertex_buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: None,
        contents: vertices_bytes,
        usage,
    });

    Model {
        bind_group,
        vertex_buffer,
        render_pipeline,
    }
}

fn view(_app: &App, model: &Model, frame: Frame) {
    let mut encoder = frame.command_encoder();
    let mut render_pass = wgpu::RenderPassBuilder::new()
        .color_attachment(frame.texture_view(), |color| color)
        .begin(&mut encoder);
    render_pass.set_bind_group(0, &model.bind_group, &[]);
    render_pass.set_pipeline(&model.render_pipeline);
    render_pass.set_vertex_buffer(0, model.vertex_buffer.slice(..));
    let vertex_range = 0..VERTICES.len() as u32;
    let instance_range = 0..1;
    render_pass.draw(vertex_range, instance_range);
}

// See the `nannou::wgpu::bytes` documentation for why this is necessary.
fn vertices_as_bytes(data: &[Vertex]) -> &[u8] {
    unsafe { wgpu::bytes::from_slice(data) }
}
