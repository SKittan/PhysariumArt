use bytemuck;
use wgpu;
use wgpu::util::DeviceExt;
use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};
use pollster;
use rand::{Rng, prelude::ThreadRng};
use serde_json;
use std::{iter, f32::consts::PI, fs};

mod gpu_create;
use gpu_create::{create_physarum_bind_group,
                 create_slime_bind_group,
                 create_bind_group_layout_compute_agents,
                 create_bind_group_layout_compute_slime,
                 create_render_bind_group,
                 create_bind_group_layout_render,
                 create_compute_pipeline, create_pipeline_layout,
                 Agent, Uniforms, Vertex};


struct State {
    surface: wgpu::Surface,
    device: wgpu::Device,
    queue: wgpu::Queue,
    render_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    slime_agents: wgpu::Buffer,
    slime_slime: wgpu::Buffer,
    slime_size: wgpu::BufferAddress,
    bind_group_physarum: wgpu::BindGroup,
    bind_group_slime: wgpu::BindGroup,
    bind_group_r: wgpu::BindGroup,
    compute_physarum: wgpu::ComputePipeline,
    compute_slime: wgpu::ComputePipeline,
}


// The vertices that make up the rectangle to which the image will be drawn.
const VERTICES: &[Vertex] = &[
    Vertex {
        position: [-1.0, -1.0],
    },
    Vertex {
        position: [1.0, -1.0],
    },
    Vertex {
        position: [1.0, 1.0],
    },
    Vertex {
        position: [-1.0, 1.0],
    },
];
const INDICES: &[u16] = &[0, 1, 2, 0, 2, 3];

fn main() {
    pollster::block_on(run());
}

impl State {
    async fn new(window: &Window) -> Self {
        let mut rng = rand::thread_rng();

        // Parameter
        const SIZE_X: u32 = 1024;
        const SIZE_Y: u32 = 1024;
        const N_AGENTS: usize = (2 as usize).pow(22);
        // init shader seeds
        let seed = rng.gen_range(1e7 as u32..9e14 as u32);

        let config_file = "./config.json";

        // Load Config from json file
        let mut cfg = Config::new(&mut rng);
        cfg.load_json(&config_file);

        window.set_inner_size(winit::dpi::PhysicalSize::new(SIZE_X, SIZE_Y));

        // The instance is a handle to our GPU
        // BackendBit::PRIMARY => Vulkan + Metal + DX12 + Browser WebGPU
        let instance = wgpu::Instance::new(wgpu::Backends::DX12);
        let surface = unsafe { instance.create_surface(window) };
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    features: wgpu::Features::empty(),
                    limits: wgpu::Limits::default()
                },
                None, // Trace path
            )
            .await
            .unwrap();

        let srf_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface.get_supported_formats(&adapter)[0],
            width: SIZE_X,
            height: SIZE_Y,
            present_mode: wgpu::PresentMode::Fifo,
        };
        surface.configure(&device, &srf_config);

        // Compute pipeline
        let cs_desc = wgpu::include_wgsl!("../Shader/Physarum.wgsl");
        let cs_mod = device.create_shader_module(cs_desc);
        let cs_slime_di_desc = wgpu::include_wgsl!("../Shader/Slime.wgsl");
        let cs_slime_di_mod = device.create_shader_module(cs_slime_di_desc);
        // Buffer for physarum agents
        // x, y, phi, 3*sensor (bool) as u32 since bool not supported
        let mut agents_init: Vec<Agent> = Vec::with_capacity(N_AGENTS);
        let c_x = SIZE_X as f32 * 0.5;
        let c_y = SIZE_Y as f32 * 0.5;
        for _ in 0 .. N_AGENTS {
            let r = rng.gen_range(0. .. cfg.r_init);
            let phi = rng.gen_range(0. .. 2.*PI);
            agents_init.push(
                Agent{
                    x: c_x + r*f32::cos(phi),
                    y: c_y + r*f32::sin(phi),
                    phi: rng.gen_range(0. .. 2.*PI)
                }
            );
        }
        let agents = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Physarum Agents"),
                contents: bytemuck::cast_slice::<_, u8>(&agents_init),
                usage:  wgpu::BufferUsages::STORAGE,
            });

        // Buffer for slime concentration
        const XY_SIZE: usize = (SIZE_X * SIZE_Y) as usize;
        let slime_size = (XY_SIZE * std::mem::size_of::<f32>())
                        as wgpu::BufferAddress;
        let slime_agents = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("SLIME Agents"),
            size: slime_size,
            usage:  wgpu::BufferUsages::STORAGE |
                    wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let slime_slime = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("SLIME Render"),
            size: slime_size,
            usage:  wgpu::BufferUsages::STORAGE |
                    wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // Fixed slime zones -> nutriment
        let mut nutriment_init: Vec<f32> = vec![0.; XY_SIZE];
        for _ in 0 .. cfg.n_fix {
            let r = rng.gen_range(cfg.r_fix_min .. cfg.r_fix_max);
            let c_x = rng.gen_range(r .. SIZE_X - r);
            let c_y = rng.gen_range(r .. SIZE_Y - r);
            for x in c_x-r .. c_x+r {
                for y in c_y-r .. c_y+r {
                    let idx = (x + y*SIZE_X) as usize;
                    nutriment_init[idx] =
                        (1. - f32::sqrt(f32::powf(x as f32 - c_x as f32, 2.) +
                                        f32::powf(y as f32 - c_y as f32, 2.))
                                        / r as f32).max(0.);
                }}
        }
        let nutriment = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Nutriment"),
                contents: bytemuck::cast_slice::<_, u8>(&nutriment_init),
                usage:  wgpu::BufferUsages::STORAGE,
            });


        // Buffer for parameter
        let uniforms = vec![Uniforms {n_agents: N_AGENTS as u32,
                                      size_x: SIZE_X, size_y: SIZE_Y,
                                      deposit: cfg.deposit, decay: cfg.decay,
                                      v: cfg.v,
                                      phi_sens: cfg.phi_sens,
                                      turn_speed: cfg.turn_speed,
                                      sens_range_min: cfg.sens_range_min,
                                      sens_range_max: cfg.sens_range_max,
                                      sense_steps: cfg.sens_range_max -
                                                   cfg.sens_range_min + 1.,
                                      seed}];
        let usage = wgpu::BufferUsages::UNIFORM;
        let uniform_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("uniform-buffer"),
                contents: bytemuck::cast_slice::<_, u8>(&uniforms),
                usage,
            });

        // Compute Pipelines //
        //____________________//
        // Physarum
        let bind_group_layout_physarum =
            create_bind_group_layout_compute_agents(&device);
        let bind_group_physarum = create_physarum_bind_group(
            &device,
            &bind_group_layout_physarum,
            &agents,
            &slime_agents,
            &slime_slime,
            &nutriment,
            &uniform_buffer);
        let pipeline_layout_physarum = create_pipeline_layout(
            &device, &bind_group_layout_physarum, "Physarum Compute");
        let physarum_pipeline = create_compute_pipeline(&device,
                                                        &pipeline_layout_physarum,
                                                        &cs_mod,
                                                        "Physarum Pipeline");
        // Slime
        // dissipation and decay
        let bind_group_layout_slime =
            create_bind_group_layout_compute_slime(&device);
        let bind_group_slime = create_slime_bind_group(&device,
                                                       &bind_group_layout_slime,
                                                       &slime_slime,
                                                       &slime_agents,
                                                       &uniform_buffer);
        let pipeline_layout_slime = create_pipeline_layout(
            &device, &bind_group_layout_slime, "Slime Layout");
        let slime_pipeline = create_compute_pipeline(
            &device, &pipeline_layout_slime, &cs_slime_di_mod,
            "Slime dissipation Pipeline");

        // Shader for Render Pipeline
        let vs_desc = wgpu::include_wgsl!("../Shader/passThrough.wgsl");
        let vs_mod = device.create_shader_module(vs_desc);
        let fs_desc = wgpu::include_wgsl!("../Shader/render.wgsl");
        let fs_mod = device.create_shader_module(fs_desc);

        let bind_group_layout_r = create_bind_group_layout_render(&device);
        let bind_group_r = create_render_bind_group(
            &device,
            &bind_group_layout_r,
            &slime_agents,
            &uniform_buffer);
        let pipeline_layout_r = create_pipeline_layout(
            &device,
            &bind_group_layout_r,
            "Physarum Render");

        let render_pipeline = device.create_render_pipeline(
            &wgpu::RenderPipelineDescriptor {
                label: Some("Render Pipeline"),
                layout: Some(&pipeline_layout_r),
                vertex: wgpu::VertexState {
                    module: &vs_mod,
                    entry_point: "main",
                    buffers: &[Vertex::desc()],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &fs_mod,
                    entry_point: "main",
                    targets: &[Some(wgpu::ColorTargetState {
                        format: srf_config.format,
                        blend: Some(wgpu::BlendState {
                            color: wgpu::BlendComponent::REPLACE,
                            alpha: wgpu::BlendComponent::REPLACE,
                        }),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: Some(wgpu::Face::Back),
                    // Setting this to anything other than Fill requires Features::POLYGON_MODE_LINE
                    // or Features::POLYGON_MODE_POINT
                    polygon_mode: wgpu::PolygonMode::Fill,
                    // Requires Features::DEPTH_CLIP_CONTROL
                    unclipped_depth: false,
                    // Requires Features::CONSERVATIVE_RASTERIZATION
                    conservative: false,
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState {
                    count: 1,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                // If the pipeline will be used with a multiview render pass, this
                // indicates how many array layers the attachments will have.
                multiview: None,
            });

        let vertex_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Vertex Buffer"),
                contents: bytemuck::cast_slice(VERTICES),
                usage: wgpu::BufferUsages::VERTEX,
            }
        );
        let index_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Index Buffer"),
                contents: bytemuck::cast_slice(INDICES),
                usage: wgpu::BufferUsages::INDEX,
            }
        );

        Self {
            surface,
            device,
            queue,
            render_pipeline,
            vertex_buffer,
            index_buffer,
            slime_agents,
            slime_slime,
            slime_size,
            bind_group_physarum,
            bind_group_slime,
            bind_group_r,
            compute_physarum: physarum_pipeline,
            compute_slime: slime_pipeline,
        }
    }

    #[allow(unused_variables)]
    fn input(&mut self, event: &WindowEvent) -> bool {
        false
    }

    fn update(&mut self) {}

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());


        // Compute pass
        let ce_desc = wgpu::CommandEncoderDescriptor {
            label: Some("Command Encoder")
        };
        let mut encoder = self.device.create_command_encoder(&ce_desc);
        {
            let c_p_pass_desc = wgpu::ComputePassDescriptor {
                label: Some("Physarum Compute Pass")
            };
            let mut c_p_pass = encoder.begin_compute_pass(&c_p_pass_desc);
            c_p_pass.set_pipeline(&self.compute_physarum);
            c_p_pass.set_bind_group(0, &self.bind_group_physarum, &[]);
            c_p_pass.dispatch_workgroups(256, 1, 1);
        }
        {
            let c_s_pass_desc = wgpu::ComputePassDescriptor {
                label: Some("Slime Pass")
            };
            let mut c_s_pass = encoder.begin_compute_pass(&c_s_pass_desc);
            c_s_pass.set_pipeline(&self.compute_slime);
            c_s_pass.set_bind_group(0, &self.bind_group_slime, &[]);
            c_s_pass.dispatch_workgroups(256, 1, 1);
        }

        // Update slime_slime for next agent step
        encoder.copy_buffer_to_buffer(&self.slime_agents, 0,
                                      &self.slime_slime, 0,
                                      self.slime_size);

        // Render pass
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0,
                            g: 0.0,
                            b: 0.0,
                            a: 1.0,
                        }),
                        store: false,
                    },
                })],
                depth_stencil_attachment: None,
            });
            render_pass.set_bind_group(0, &self.bind_group_r, &[]);
            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.set_index_buffer(self.index_buffer.slice(..),
                                         wgpu::IndexFormat::Uint16);
            render_pass.draw_indexed(0..6, 0, 0..1);
        }
        self.queue.submit(iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}

pub async fn run() {
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new().build(&event_loop).unwrap();

    // State::new uses async code, so we're going to wait for it to finish
    let mut state = State::new(&window).await;

    event_loop.run(move |event, _, control_flow| {
        match event {
            Event::WindowEvent {
                ref event,
                window_id,
            } if window_id == window.id() => {
                if !state.input(event) {
                    // UPDATED!
                    match event {
                        WindowEvent::CloseRequested
                        | WindowEvent::KeyboardInput {
                            input:
                                KeyboardInput {
                                    state: ElementState::Pressed,
                                    virtual_keycode: Some(VirtualKeyCode::Escape),
                                    ..
                                },
                            ..
                        } => *control_flow = ControlFlow::Exit,
                        _ => {}
                    }
                }
            }
            Event::RedrawRequested(window_id) if window_id == window.id() => {
                state.update();
                match state.render() {
                    Ok(_) => {}
                    // Reconfigure the surface if it's lost or outdated
                    Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated)
                    => {},
                    // The system is out of memory, we should probably quit
                    Err(wgpu::SurfaceError::OutOfMemory)
                    => *control_flow = ControlFlow::Exit,

                    Err(wgpu::SurfaceError::Timeout)
                    => println!("Surface timeout"),
                }
            }
            Event::RedrawEventsCleared => {
                // RedrawRequested will only trigger once, unless we manually
                // request it.
                window.request_redraw();
            }
            _ => {}
        }
    });
}

struct Config {
    r_init: f32,  // Radius for agent initialisation
    deposit: f32,  // Slime deposition of each agent per step
    decay: f32,
    v: f32,
    phi_sens: f32,  // Sensor angle
    turn_speed: f32,  // turn speed in rad per step
    sens_range_min: f32,
    sens_range_max: f32,
    n_fix: u32,  // Number of fixed max slime zones
    r_fix_min: u32,  // min radius of fixed max slime zones
    r_fix_max: u32  // max radius of fixed max slime zones
}

impl Config {
    fn new(rng: &mut ThreadRng) -> Config {
        Config {
            r_init: rng.gen_range(5. .. 100.),
            deposit: rng.gen_range(0.0001 .. 0.1),
            decay: rng.gen_range(0.1 .. 0.9),
            v: rng.gen_range(0.5 .. 25.),
            phi_sens: rng.gen_range(0.1*PI .. 0.5*PI),
            turn_speed: rng.gen_range(0.01 .. 0.2*PI),
            sens_range_min: rng.gen_range(1. .. 5.),
            sens_range_max: rng.gen_range(5. .. 50.),
            n_fix: rng.gen_range(0 .. 25),
            r_fix_min: rng.gen_range(1 .. 2),
            r_fix_max: rng.gen_range(2 .. 10)
        }
    }

    fn load_json(&mut self, config_file: &str) {
        let data = fs::read_to_string(config_file);
        let data = match data {
            Ok(data) => data,
            Err(_) => {
                println!("Error open config file: {:?}", config_file);
                println!("Init with random configuration.");
                self.show_state();
                "".to_string()
            }
        };
        if data != "" {
            let json_res: Result<serde_json::Value, serde_json::Error> =
                serde_json::from_str(&data);
            match json_res {
                Ok(json) => {
                    self.r_init = json["r_init"].as_f64().unwrap() as f32;
                    self.deposit = json["deposit"].as_f64().unwrap() as f32;
                    self.decay = json["decay"].as_f64().unwrap() as f32;
                    self.v = json["v"].as_f64().unwrap() as f32;
                    self.phi_sens =
                        json["phi_sens"].as_f64().unwrap() as f32;
                    self.turn_speed =
                        json["turn_speed"].as_f64().unwrap() as f32;
                    self.sens_range_min =
                        json["sens_range_min"].as_f64().unwrap() as f32;
                    self.sens_range_max =
                        json["sens_range_max"].as_f64().unwrap() as f32;
                    self.n_fix =
                        json["n_fix"].as_u64().unwrap() as u32;
                    self.r_fix_min =
                        json["r_fix_min"].as_u64().unwrap() as u32;
                    self.r_fix_max =
                        json["r_fix_max"].as_u64().unwrap() as u32;
                },
                Err(e) => {
                    println!("Error reading config: {:?}", e);
                    println!("Init with random configuration.");
                    self.show_state();
                }
            };
        }
    }

    fn show_state(&self) {
        println!("Physarum configuration:\n--");
        println!("  r_init: {:?}", self.r_init);
        println!("  deposit: {:?}", self.deposit);
        println!("  decay: {:?}", self.decay);
        println!("  v: {:?}", self.v);
        println!("  phi_sens: {:?}", self.phi_sens);
        println!("  turn_speed: {:?}", self.turn_speed);
        println!("  sens_range_min: {:?}", self.sens_range_min);
        println!("  sens_range_max: {:?}", self.sens_range_max);
        println!("  n_fix: {:?}", self.n_fix);
        println!("  r_fix_min: {:?}", self.r_fix_min);
        println!("  r_fix_max: {:?}", self.r_fix_max);
    }

}