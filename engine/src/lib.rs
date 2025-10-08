pub fn start() {}

pub trait Field {}

pub enum BrushOperation {
    Paint,
}

pub struct Brush {
    field: Box<dyn Field>,
    op: BrushOperation,
    block: Block,
}

pub struct Block(usize);

pub struct Structure(usize);

pub trait Volume {
    fn get();
    fn set();
}

pub trait Voxel {
    fn apply(&self, brush: Brush) {}
}

pub trait Graphics {
    fn hide();
    fn show();
}

pub trait Physics {
    fn force();
    fn torque();
}

 
