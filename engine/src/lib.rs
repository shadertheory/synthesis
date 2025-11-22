#![feature(
    trait_alias,
    thread_local,
    inherent_associated_types,
    generic_const_exprs,
    associated_type_defaults,
    ptr_metadata,
    box_as_ptr,
    type_alias_impl_trait,
    random
)]

mod acoustics;
mod darwin;
mod world;

use math::*;
use std::{
    cell::{RefCell, UnsafeCell},
    collections::BTreeMap,
    convert::identity,
    fmt,
    iter::{self, Sum},
    mem::{self, MaybeUninit},
    ops::{BitAnd, BitOr, BitOrAssign, Deref, Index, IndexMut, Neg, Shl, Shr, *},
    slice,
};

#[thread_local]
pub static mut INSTANCE: Option<Platform> = None;

type Platform = darwin::Darwin;

#[unsafe(no_mangle)]
pub extern "C" fn engine_start(data: *const ()) {
    unsafe {
        INSTANCE = Some(Platform::init(data));
        #[allow(static_mut_refs)]
        let instance = INSTANCE.as_mut().unwrap();
    };
}

#[unsafe(no_mangle)]
pub extern "C" fn engine_draw() {
    unsafe {
        #[allow(static_mut_refs)]
        INSTANCE.as_mut().unwrap().draw();
    }
}

pub trait Renderer {
    fn hide();
    fn show();
}

pub trait Rigidbody {
    fn force();
    fn torque();
}
