use nannou::prelude::*;

mod gpu_create;
use gpu_create::{create_bind_group, create_bind_group_layout_compute,
                 create_bind_group_layout_render,
                 create_compute_pipeline, create_render_pipeline,
                 create_pipeline_layout,
                 Physarum, Uniforms, Vertex};


struct Model {
    bind_group_r: wgpu::BindGroup,
    render_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    physarum: Physarum
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

    // Compute pipeline
    let cs_mod =
    wgpu::shader_from_spirv_bytes(
        device, include_bytes!("../Shader/Physarum.comp.spv"));
    // Buffer for slime concentration
    let slime_size =
        ((size_x*size_y) as usize *
        std::mem::size_of::<f32>()) as wgpu::BufferAddress;
    let slime = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("SLIME"),
        size: slime_size,
        usage: wgpu::BufferUsage::STORAGE |
               wgpu::BufferUsage::COPY_DST |
               wgpu::BufferUsage::COPY_SRC,
        mapped_at_creation: false,
    });

    // Buffer for parameter
    let uniforms = Uniforms {n_agents: 100, size_x, size_y};
    let uniforms_bytes = uniforms_as_bytes(&uniforms);
    let usage = wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST;
    let uniform_buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("uniform-buffer"),
        contents: uniforms_bytes,
        usage,
    });

    let bind_group_layout_c = create_bind_group_layout_compute(device);
    let bind_group_c = create_bind_group(device,
                                         &bind_group_layout_c,
                                         &slime, slime_size,
                                         &uniform_buffer);
    let pipeline_layout_c = create_pipeline_layout(&device,
                                                   &bind_group_layout_c);
    let compute_pipeline = create_compute_pipeline(&device,
                                                   &pipeline_layout_c,
                                                   &cs_mod);

    // Render Pipeline
    let vs_mod =
        wgpu::shader_from_spirv_bytes
            (device, include_bytes!("../Shader/passThrough.vert.spv"));
    let fs_mod =
        wgpu::shader_from_spirv_bytes(
            device, include_bytes!("../Shader/render.frag.spv"));

    let bind_group_layout_r =
        create_bind_group_layout_render(device);
    let bind_group_r = create_bind_group(device,
                                         &bind_group_layout_r,
                                         &slime, slime_size,
                                         &uniform_buffer);
    let pipeline_layout_r = create_pipeline_layout(device,
                                                   &bind_group_layout_r);
    let render_pipeline = create_render_pipeline(
        device,
        &pipeline_layout_r,
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

    let physarum = Physarum {
        slime,
        slime_size,
        uniform_buffer,
        bind_group: bind_group_c,
        pipeline: compute_pipeline
    };

    Model {
        bind_group_r,
        vertex_buffer,
        render_pipeline,
        physarum
    }
}

fn view(app: &App, model: &Model, frame: Frame) {
    let window = app.main_window();
    let device = window.swap_chain_device();
    // Compute pass
    let c_desc = wgpu::CommandEncoderDescriptor {
        label: Some("Physarun Compute Encoder")
    };
    let mut c_encoder = device.create_command_encoder(&c_desc);
    {
        let c_pass_desc = wgpu::ComputePassDescriptor {
            label: Some("Physarum Compute Pass")
        };
        let mut c_pass = c_encoder.begin_compute_pass(&c_pass_desc);
        c_pass.set_pipeline(&model.physarum.pipeline);
        c_pass.set_bind_group(0, &model.physarum.bind_group, &[]);
        c_pass.dispatch(256, 1, 1);
    }
    window.swap_chain_queue().submit(Some(c_encoder.finish()));

    // Render pass
    let mut r_encoder = frame.command_encoder();
    let mut render_pass = wgpu::RenderPassBuilder::new()
        .color_attachment(frame.texture_view(), |color| color)
        .begin(&mut r_encoder);
    render_pass.set_bind_group(0, &model.bind_group_r, &[]);
    render_pass.set_pipeline(&model.render_pipeline);
    render_pass.set_vertex_buffer(0, model.vertex_buffer.slice(..));
    let vertex_range = 0..VERTICES.len() as u32;
    let instance_range = 0..1;
    render_pass.draw(vertex_range, instance_range);
}

// See the `nannou::wgpu::bytes` documentation for why this is necessary.
fn uniforms_as_bytes(uniforms: &Uniforms) -> &[u8] {
    unsafe { wgpu::bytes::from(uniforms) }
}

fn vertices_as_bytes(data: &[Vertex]) -> &[u8] {
    unsafe { wgpu::bytes::from_slice(data) }
}
