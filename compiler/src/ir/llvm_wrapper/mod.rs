pub mod execution_engine;
pub mod ir_components;
pub mod jit;

#[cfg(test)]
mod test;

use anyhow::Result;

use std::{
    ffi::{CStr, CString},
    marker::PhantomData,
};

use paste::paste;

use llvm_sys::{
    core::{
        LLVMBuildAdd, LLVMBuildRet, LLVMContextCreate, LLVMContextDispose,
        LLVMCreateBuilderInContext, LLVMDisposeBuilder, LLVMDisposeModule, LLVMDumpModule,
        LLVMModuleCreateWithName, LLVMPositionBuilderAtEnd,
    },
    prelude::*,
};

use crate::ir::llvm_wrapper::{
    execution_engine::ExecutionEngine,
    ir_components::{
        LLVMBasicBlock, LLVMFunction, LLVMFunctionType, LLVMTypeInContext, LLVMValue,
        TypeRefBuilder,
    },
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

impl LLVMContext {
    pub fn make_builder<'c>(&'c self) -> LLVMBuilder<'c> {
        LLVMBuilder::new(self)
    }
    pub fn make_type<'ctx>(&'ctx self, builder: TypeRefBuilder) -> LLVMTypeInContext<'ctx> {
        LLVMTypeInContext::new(self, builder)
    }

    pub fn make_bb<'c, 'm>(
        &'c self,
        function: &LLVMFunction<'c, 'm>,
        name: &CStr,
    ) -> LLVMBasicBlock<'c> {
        LLVMBasicBlock::new(self, function, name)
    }
}

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

impl LLVMModule {
    pub fn make_function_val<'c, 'm>(
        &'m self,
        name: &CStr,
        function_type: LLVMFunctionType<'c>,
    ) -> LLVMFunction<'c, 'm> {
        LLVMFunction::new(self, name, function_type)
    }

    pub fn dump(&self) {
        unsafe {
            LLVMDumpModule(self.module_ref);
        }
    }

    pub fn make_execution_engine<'m>(&'m self) -> Result<ExecutionEngine<'m>> {
        ExecutionEngine::new(self)
    }
}

pub struct LLVMBuilder<'ctx> {
    builder_ref: LLVMBuilderRef,
    phantom_data: PhantomData<&'ctx LLVMContext>,
}

impl<'c> LLVMBuilder<'c> {
    pub fn new(context: &'c LLVMContext) -> Self {
        Self {
            builder_ref: unsafe { LLVMCreateBuilderInContext(context.context_ref) },
            phantom_data: PhantomData,
        }
    }

    pub fn move_to_end(&self, bb: &LLVMBasicBlock<'c>) {
        unsafe {
            LLVMPositionBuilderAtEnd(self.builder_ref, bb.basic_block_ref);
        };
    }

    pub fn ret<'m>(&'c self, value: LLVMValue<'c, 'm>) -> LLVMValue<'c, 'm> {
        LLVMValue::new(unsafe { LLVMBuildRet(self.builder_ref, value.value_ref) })
    }

    pub fn add<'m>(
        &self,
        a: LLVMValue<'c, 'm>,
        b: LLVMValue<'c, 'm>,
        name: &CStr,
    ) -> LLVMValue<'c, 'm> {
        LLVMValue::new(unsafe {
            LLVMBuildAdd(
                self.builder_ref,
                a.value_ref,
                b.value_ref,
                name.as_ptr() as *const _,
            )
        })
    }
}

impl<'a> Drop for LLVMBuilder<'a> {
    fn drop(&mut self) {
        unsafe {
            LLVMDisposeBuilder(self.builder_ref);
        }
    }
}
