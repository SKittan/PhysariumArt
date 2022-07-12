fn model(app: &App) -> Model {
    let (size_x, size_y) = (512, 512);
    let n_agents: usize = 300;
    let decay: f32 = 0.9;

    let mut rng = rand::thread_rng();

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
    let mut agents_init: Vec<Agent> = Vec::with_capacity(n_agents);
    for _ in 0 .. n_agents {
        agents_init.push(
            Agent{
                x: rng.gen_range(0. .. size_x as f32),
                y: rng.gen_range(0. .. size_y as f32),
                phi: rng.gen_range(-3.14 .. 3.14),
                sens: 0
            }
        );
    }
    let agents = device.create_buffer_init(&wgpu::BufferInitDescriptor {
        label: Some("Physarum Agents"),
        contents: bytemuck::cast_slice::<_, u8>(&agents_init),
        usage:  wgpu::BufferUsages::STORAGE,
    });

    // Buffer for slime concentration
    let xy_size: usize = (size_x * size_y) as usize;
    let slime_size = (xy_size * std::mem::size_of::<f32>())
                     as wgpu::BufferAddress;
    let slime_agents = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("SLIME Agents"),
        size: slime_size,
        usage:  wgpu::BufferUsages::STORAGE |
                wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let slime_slime = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("SLIME Render"),
        size: slime_size,
        usage:  wgpu::BufferUsages::STORAGE |
                wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // Buffer for parameter
    let uniforms = vec![Uniforms {n_agents: n_agents as u32,
                                  size_x, size_y, decay}];
    let usage = wgpu::BufferUsages::UNIFORM;
    let uniform_buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("uniform-buffer"),
        contents: bytemuck::cast_slice::<_, u8>(&uniforms),
        usage,
    });

    // Compute Pipelines //
    //____________________//
    // Physarum
    let bind_group_layout_physarum = create_bind_group_layout_compute(device,
                                                                      false);
    let bind_group_physarum = create_physarum_bind_group(
        device, &bind_group_layout_physarum, &agents, &n_agents,
        &slime_agents, &xy_size, &uniform_buffer);
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
                                                   &slime_agents, &slime_slime,
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
                                                &slime_agents, &xy_size,
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
        slime_agents,
        slime_slime,
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
        c_p_pass.dispatch(256, 1, 1);
    }
    {
        let c_s_pass_desc = wgpu::ComputePassDescriptor {
            label: Some("Slime Pass")
        };
        let mut c_s_pass = encoder.begin_compute_pass(&c_s_pass_desc);
        c_s_pass.set_pipeline(&model.physarum.compute_slime);
        c_s_pass.set_bind_group(0, &model.physarum.bind_group_slime, &[]);
        c_s_pass.dispatch(256, 1, 1);
    }

    encoder.copy_buffer_to_buffer(&model.physarum.slime_slime, 0,
                                  &model.physarum.slime_agents, 0,
                                  model.physarum.slime_size);
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