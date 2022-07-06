use nannou::{prelude::*};

mod gpu_create;
use gpu_create::{create_physarum_bind_group,
                 create_slime_bind_group,
                 create_bind_group_layout_compute,
                 create_render_bind_group,
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
    let n_agents: usize = 10;//24;
    let decay: f32 = 0.9;

    let w_id = app
        .new_window()
        .size(size_x, size_y)
        .view(view)
        .build()
        .unwrap();
    let window = app.window(w_id).unwrap();
    let device = window.device();
    let format = Frame::TEXTURE_FORMAT;
    let msaa_samples = window.msaa_samples();

    // Compute pipeline
    let cs_desc = wgpu::include_wgsl!("../Shader/Physarum.wgsl");
    let cs_mod = device.create_shader_module(&cs_desc);
    let cs_slime_di_desc = wgpu::include_wgsl!("../Shader/Slime.wgsl");
    let cs_slime_di_mod = device.create_shader_module(&cs_slime_di_desc);
    // Buffer for physarum agents
    // x, y, phi, 3*sensor (bool) as u32 since bool not supported
    let agent_size = ((3 * std::mem::size_of::<f32>() +
                       std::mem::size_of::<u32>()) * n_agents)
                     as wgpu::BufferAddress;
    let agents = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Physarum Agents"),
        size: agent_size,
        usage:  wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    // Buffer for slime concentration
    let xy_size: usize = (size_x * size_y) as usize;
    let slime_size = (xy_size * std::mem::size_of::<f32>())
                     as wgpu::BufferAddress;
    let slime_agents = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("SLIME Agents"),
        size: slime_size,
        usage:  wgpu::BufferUsages::STORAGE |
                wgpu::BufferUsages::COPY_SRC |
                wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let slime_in = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("SLIME IN"),
        size: slime_size,
        usage:  wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let slime_out = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("SLIME OUT"),
        size: slime_size,
        usage:  wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let slime_render = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("SLIME Render"),
        size: slime_size,
        usage:  wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Buffer for parameter
    let uniforms = Uniforms {n_agents, size_x, size_y, decay};
    let uniforms_bytes = uniforms_as_bytes(&uniforms);
    let usage = wgpu::BufferUsages::UNIFORM;
    let uniform_buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("uniform-buffer"),
        contents: uniforms_bytes,
        usage,
    });

    // Compute Pipelines //
    //____________________//
    // Physarum
    let bind_group_layout_physarum = create_bind_group_layout_compute(device,
                                                                      false);
    let bind_group_physarum = create_physarum_bind_group(
        device, &bind_group_layout_physarum, &agents, agent_size,
        &slime_agents, &uniform_buffer);
    let pipeline_layout_physarum = create_pipeline_layout(
        &device, &bind_group_layout_physarum, "Physarum Compute");
    let physarum_pipeline = create_compute_pipeline(&device,
                                                    &pipeline_layout_physarum,
                                                    &cs_mod,
                                                    "Physarum Pipeline");
    // Slime
    // dissipation and decay
    let bind_group_layout_slime = create_bind_group_layout_compute(device,
                                                                   true);
    let bind_group_slime = create_slime_bind_group(device,
                                                   &bind_group_layout_slime,
                                                   &slime_in, &slime_out,
                                                   &xy_size,
                                                   &uniform_buffer);
    let pipeline_layout_slime = create_pipeline_layout(
        &device, &bind_group_layout_slime, "Slime Layout");
    let slime_pipeline = create_compute_pipeline(
        &device, &pipeline_layout_slime, &cs_slime_di_mod,
        "Slime dissipation Pipeline");

    // Render Pipeline
    let vs_desc = wgpu::include_wgsl!("../Shader/passThrough.wgsl");
    let vs_mod = device.create_shader_module(&vs_desc);
    let fs_desc = wgpu::include_wgsl!("../Shader/render.wgsl");
    let fs_mod = device.create_shader_module(&fs_desc);

    let bind_group_layout_r =
        create_bind_group_layout_render(device);
    let bind_group_r = create_render_bind_group(device,
                                                &bind_group_layout_r,
                                                &slime_render, slime_size,
                                                &uniform_buffer);
    let pipeline_layout_r = create_pipeline_layout(device,
                                                   &bind_group_layout_r,
                                                   "Physarum Render");
    let render_pipeline = create_render_pipeline(
        device,
        &pipeline_layout_r,
        &vs_mod,
        &fs_mod,
        format,
        msaa_samples,
    );

    let vertices_bytes = vertices_as_bytes(&VERTICES[..]);
    let usage = wgpu::BufferUsages::VERTEX;
    let vertex_buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: None,
        contents: vertices_bytes,
        usage,
    });

    let physarum = Physarum {
        agents,
        agent_size,
        slime_agents,
        slime_in,
        slime_out,
        slime_render,
        slime_size,
        uniform_buffer,
        bind_group_physarum,
        bind_group_slime,
        compute_physarum: physarum_pipeline,
        compute_slime: slime_pipeline,
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
    let device = window.device();

    // Compute pass
    let ce_desc = wgpu::CommandEncoderDescriptor {
        label: Some("Command Encoder")
    };
    let mut encoder = device.create_command_encoder(&ce_desc);
    {
        let c_p_pass_desc = wgpu::ComputePassDescriptor {
            label: Some("Physarum Compute Pass")
        };
        let mut c_p_pass = encoder.begin_compute_pass(&c_p_pass_desc);
        c_p_pass.set_pipeline(&model.physarum.compute_physarum);
        c_p_pass.set_bind_group(0, &model.physarum.bind_group_physarum, &[]);
        c_p_pass.dispatch(10, 1, 1);
    }
    {
        let c_s_pass_desc = wgpu::ComputePassDescriptor {
            label: Some("Slime Pass")
        };
        let mut c_s_pass = encoder.begin_compute_pass(&c_s_pass_desc);
        c_s_pass.set_pipeline(&model.physarum.compute_slime);
        c_s_pass.set_bind_group(0, &model.physarum.bind_group_slime, &[]);
        c_s_pass.dispatch(10, 1, 1);
    }
    window.queue().submit(Some(encoder.finish()));

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
