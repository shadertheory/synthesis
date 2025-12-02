pub trait Component {}

mod acoustic {
    #[derive(Component)]
    pub struct Source(u64);
}

mod agent {
    use bitflags::bitflags;
    bitflags! {
        #[derive(Component)]
        pub struct State: u16 {
             const SENDER = 1 << 0;
             const RECEIVER = 1 << 1;
        }
    }
}

mod transform {
    #[derive(Component)]
    pub struct Transform(Vector<3, f32>);
    #[derive(Component)]
    pub struct Scale(Vector<3, f32>);
    #[derive(Component)]
    pub struct Rotation(Quaternion);
}

mod physics {
    #[derive(Component)]
    pub struct Distance(f32);
    #[derive(Component)]
    pub struct Energy(f32);
    #[derive(Component)]
    pub struct Time(f32);
    #[derive(Component)]
    pub struct Life(f32);
    #[derive(Component)]
    pub struct Mass(f32);
    #[derive(Component)]
    pub struct Vorticity(Vector<3, f32>);
    #[derive(Component)]
    pub struct Direction(Vector<3, f32>);
    #[derive(Component)]
    pub struct Pressure(f32);
    #[derive(Component)]
    pub struct Density(f32);
}

mod enemy {
    #[derive(Component)]
    pub struct Aggression(f32);
    #[derive(Component)]
    pub struct Target;
    #[derive(Component)]
    pub struct Flock(Vector<3, f32>);
}

mod sensor {
    pub trait Sensor: Component {
        fn receives() -> Particle;
        fn sends() -> Option<Particle> {
            None
        }
    }

    pub enum Lens {
        Perspective { fov: f32 },
    }

    #[derive(Sensor)]
    pub struct Camera {
        near: f32,
        far: f32,
        aspect: f32,
        lens: Lens,
    }

    impl Sensor for Camera {
        fn receives() -> Particle {
            Particle::PHOTON
        }
    }

    #[derive(Sensor)]
    pub struct Microphone {
        gain: f32,
    }

    impl Sensor for Microphone {
        fn receives() -> Particle {
            Particle::PHONON
        }
        fn sends() -> Option<Particle> {
            Particle::PHONON
        }
    }

    use bitflags::bitflags;
    bitflags! {
        #[derive(Component)]
        /* All particles fundamentally carry translation, rotation, direction, energy, distance */
        pub struct Particle: u16 {
            /*
                A light particle. Sends information.
                Must carry fundamentals.
                Through the engine will output directly to the framebuffer with color.
            */
            const PHOTON = 1 << 0;
            /*
                A sound particle. Both, sends & receives information.
                Sending:
                    Must carry fundamentals, as well as phase and life.
                    Used to probe the environment for its acoustics,
                                as well as diffractive audio.
                Receiving:
                    Used to probe for occlusion between the microphone and the source.
                Through the engine will coordinate directly with the active sound output.
            */
            const PHONON = 1 << 1;
           /*
                A rigidbody particle. Sends information.
                Must carry fundamentals, as well as mass.
            */
            const STEREON = 1 << 2;
            //fluid TODO
            const RHENON = 1 << 3;

            const PHYSICAL = Self::STEREON | Self::RHENON;
        }
    }

    //Denotes the fundamental agent type this collider detects.
    //For instance, stereon and rhenon are physical collider triggers.
    //Intelligent zombie agents may use photon and phonon triggers for their AI.
    pub type Trigger = Particle;

    #[derive(Component)]
    pub struct Collider {
        trigger: Trigger,
        solid: bool,
    }

    impl Sensor for Collider {
        fn receives() -> Particle {
            Particle::PHYSICAL
        }
    }
}

mod scene {
    pub struct Geometry(Option<u64>);
    pub struct Instance(Option<u64>);

    pub struct Form {
        min: Vector<i64, 3>,
        max: Vector<i64, 3>,
    }
}

pub struct World {}

pub struct SparseMap {}
