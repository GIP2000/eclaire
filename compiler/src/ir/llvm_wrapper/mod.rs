pub mod ir_components;
pub mod jit;

#[cfg(test)]
mod test;

use std::{ffi::CString, marker::PhantomData};

use paste::paste;

use llvm_sys::{
    core::{
        LLVMContextCreate, LLVMContextDispose, LLVMCreateBuilderInContext, LLVMDisposeBuilder,
        LLVMDisposeModule, LLVMModuleCreateWithName,
    },
    prelude::*,
};

macro_rules! create_unsafe_warpper {
    ($name:ident, $ref_type: tt, $context_builder: expr, $drop_call: ident) => {
        paste! {
            pub struct [<LLVM $name>] {
                [<$name:lower _ref>]: $ref_type
            }

            impl [<LLVM $name>] {
                pub fn new() -> Self {
                    Self {
                        [<$name:lower _ref>]: unsafe {$context_builder}
                    }
                }
            }

            impl Drop for [<LLVM $name>] {
                fn drop(&mut self) {
                    unsafe {
                        $drop_call(self.[<$name:lower _ref>])
                    }
                }
            }

        }
    };
}

create_unsafe_warpper!(
    Context,
    LLVMContextRef,
    LLVMContextCreate(),
    LLVMContextDispose
);

create_unsafe_warpper!(
    Module,
    LLVMModuleRef,
    LLVMModuleCreateWithName(
        CString::new("eclaire_module")
            .expect("c strings should work")
            .as_ptr(),
    ),
    LLVMDisposeModule
);

pub struct LLVMBuilder<'a> {
    builder_ref: LLVMBuilderRef,
    phantom_data: PhantomData<&'a LLVMContext>,
}

impl<'a> LLVMBuilder<'a> {
    pub fn new(context: &'a mut LLVMContext) -> Self {
        Self {
            builder_ref: unsafe { LLVMCreateBuilderInContext(context.context_ref) },
            phantom_data: PhantomData,
        }
    }
}

impl<'a> Drop for LLVMBuilder<'a> {
    fn drop(&mut self) {
        unsafe {
            LLVMDisposeBuilder(self.builder_ref);
        }
    }
}
