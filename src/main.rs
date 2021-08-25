use nannou::prelude::*;

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
    // Buffer for physarum agents
    // position + sensor, heading
    let agent_size = ((6 * std::mem::size_of::<f32>() +
                       8 * std::mem::size_of::<bool>()) * n_agents)
                     as wgpu::BufferAddress;
    let agents = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Physarum Agents"),
        size: agent_size,
        usage:  wgpu::BufferUsage::STORAGE |
                wgpu::BufferUsage::COPY_DST |
                wgpu::BufferUsage::COPY_SRC,
        mapped_at_creation: false,
    });
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
    let slime_dissipate = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("SLIME dissipation"),
        size: slime_size,
        usage: wgpu::BufferUsage::STORAGE |
               wgpu::BufferUsage::COPY_DST |
               wgpu::BufferUsage::COPY_SRC,
        mapped_at_creation: false,
    });

    // Buffer for parameter
    let uniforms = Uniforms {n_agents, size_x, size_y};
    let uniforms_bytes = uniforms_as_bytes(&uniforms);
    let usage = wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST;
    let uniform_buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("uniform-buffer"),
        contents: uniforms_bytes,
        usage,
    });

    // Cimpute Pipelines //
    //____________________//
    // Physarum
    let bind_group_layout_physarum = create_bind_group_layout_compute(device);
    let bind_group_physarum = create_physarum_bind_group(
        device, &bind_group_layout_physarum, &agents, agent_size,
        &slime, slime_size, &uniform_buffer);
    let pipeline_layout_physarum = create_pipeline_layout(
        &device, &bind_group_layout_physarum, "Physarum Compute");
    let physarum_pipeline = create_compute_pipeline(&device,
                                                    &pipeline_layout_physarum,
                                                    &cs_mod,
                                                    "Physarum Pipeline");
    // Slime
    // dissipation
    let bind_group_layout_slime = create_bind_group_layout_compute(device);
    let bind_group_slime = create_slime_bind_group(device,
                                                   &bind_group_layout_slime,
                                                   &slime, &slime_dissipate,
                                                   slime_size,
                                                   &uniform_buffer);
    let pipeline_layout_slime = create_pipeline_layout(
        &device, &bind_group_layout_slime, "Slime Layout");
    let slime_di_pipeline = create_compute_pipeline(
        &device, &pipeline_layout_slime, &cs_mod, "Slime dissipation Pipeline");
    // decay
    let slime_de_pipeline = create_compute_pipeline(
        &device, &pipeline_layout_slime, &cs_mod, "Slime decay Pipeline");

    // Render Pipeline
    let vs_mod =
        wgpu::shader_from_spirv_bytes
            (device, include_bytes!("../Shader/passThrough.vert.spv"));
    let fs_mod =
        wgpu::shader_from_spirv_bytes(
            device, include_bytes!("../Shader/render.frag.spv"));

    let bind_group_layout_r =
        create_bind_group_layout_render(device);
    let bind_group_r = create_render_bind_group(device,
                                                &bind_group_layout_r,
                                                &slime, slime_size,
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
    let usage = wgpu::BufferUsage::VERTEX;
    let vertex_buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: None,
        contents: vertices_bytes,
        usage,
    });

    let physarum = Physarum {
        agents,
        agent_size,
        slime,
        slime_size,
        uniform_buffer,
        bind_group_physarum,
        bind_group_slime,
        compute_physarum: physarum_pipeline,
        compute_dissipation: slime_di_pipeline,
        compute_decay: slime_de_pipeline
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
    let c_p_desc = wgpu::CommandEncoderDescriptor {
        label: Some("Physarun Compute Encoder")
    };
    let mut c_p_encoder = device.create_command_encoder(&c_p_desc);
    {
        let c_p_pass_desc = wgpu::ComputePassDescriptor {
            label: Some("Physarum Compute Pass")
        };
        let mut c_p_pass = c_p_encoder.begin_compute_pass(&c_p_pass_desc);
        c_p_pass.set_pipeline(&model.physarum.compute_physarum);
        c_p_pass.set_bind_group(0, &model.physarum.bind_group_physarum, &[]);
        c_p_pass.dispatch(256, 1, 1);
    }
    window.swap_chain_queue().submit(Some(c_p_encoder.finish()));

    let c_sdi_desc = wgpu::CommandEncoderDescriptor {
        label: Some("Slime Dissipation Encoder")
    };
    let mut c_sdi_encoder = device.create_command_encoder(&c_sdi_desc);
    {
        let c_sdi_pass_desc = wgpu::ComputePassDescriptor {
            label: Some("Slime Dissipation Pass")
        };
        let mut c_sdi_pass = c_sdi_encoder.begin_compute_pass(&c_sdi_pass_desc);
        c_sdi_pass.set_pipeline(&model.physarum.compute_dissipation);
        c_sdi_pass.set_bind_group(0, &model.physarum.bind_group_slime, &[]);
        c_sdi_pass.dispatch(256, 1, 1);
    }
    window.swap_chain_queue().submit(Some(c_sdi_encoder.finish()));

    let c_sde_desc = wgpu::CommandEncoderDescriptor {
        label: Some("Slime decay Encoder")
    };
    let mut c_sde_encoder = device.create_command_encoder(&c_sde_desc);
    {
        let c_sde_pass_desc = wgpu::ComputePassDescriptor {
            label: Some("Slime decay Pass")
        };
        let mut c_sde_pass = c_sde_encoder.begin_compute_pass(&c_sde_pass_desc);
        c_sde_pass.set_pipeline(&model.physarum.compute_decay);
        c_sde_pass.set_bind_group(0, &model.physarum.bind_group_slime, &[]);
        c_sde_pass.dispatch(256, 1, 1);
    }
    window.swap_chain_queue().submit(Some(c_sde_encoder.finish()));

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
