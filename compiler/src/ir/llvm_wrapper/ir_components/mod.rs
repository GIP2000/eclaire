use llvm_sys::{
    core::{LLVMAddFunction, LLVMAppendBasicBlockInContext, LLVMGetParam},
    prelude::{LLVMBasicBlockRef, LLVMContextRef, LLVMTypeRef, LLVMValueRef},
};
use std::{ffi::CStr, marker::PhantomData};

use super::{LLVMContext, LLVMModule};

pub type TypeRefBuilder = unsafe extern "C" fn(context: LLVMContextRef) -> LLVMTypeRef;

#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct LLVMTypeInContext<'c> {
    type_ref: LLVMTypeRef,
    phantom_data: PhantomData<&'c LLVMContext>,
}

impl<'c> LLVMTypeInContext<'c> {
    pub fn new(context: &'c LLVMContext, builder: TypeRefBuilder) -> Self {
        Self {
            type_ref: unsafe { builder(context.context_ref) },
            phantom_data: PhantomData,
        }
    }
}

pub struct LLVMFunctionType<'c> {
    type_in_c: LLVMTypeInContext<'c>,
}

impl<'c> LLVMFunctionType<'c> {
    pub fn new(ret: LLVMTypeInContext<'c>, args: &mut [LLVMTypeInContext]) -> Self {
        let args: &mut [LLVMTypeRef] = unsafe { std::mem::transmute(args) };

        let type_ref = unsafe {
            llvm_sys::core::LLVMFunctionType(
                ret.type_ref,
                args.as_mut_ptr(),
                args.len() as std::ffi::c_uint,
                0,
            )
        };

        Self {
            type_in_c: LLVMTypeInContext {
                type_ref,
                phantom_data: PhantomData,
            },
        }
    }
}

pub struct LLVMValue<'c, 'm> {
    pub(crate) value_ref: LLVMValueRef,
    phantom_data_c: PhantomData<&'c LLVMContext>,
    phantom_data_module: PhantomData<&'m LLVMContext>,
}
impl<'c, 'm> LLVMValue<'c, 'm> {
    pub fn new(value_ref: LLVMValueRef) -> Self {
        Self {
            value_ref,
            phantom_data_c: PhantomData,
            phantom_data_module: PhantomData,
        }
    }
}

pub struct LLVMFunction<'c, 'm> {
    value: LLVMValue<'c, 'm>,
}

impl<'c, 'm> LLVMFunction<'c, 'm> {
    pub fn new(module: &'m LLVMModule, name: &CStr, function_type: LLVMFunctionType<'c>) -> Self {
        let value_ref = unsafe {
            LLVMAddFunction(
                module.module_ref,
                name.as_ptr() as *const _,
                function_type.type_in_c.type_ref,
            )
        };
        Self {
            value: LLVMValue::new(value_ref),
        }
    }

    pub fn get_arg(&self, idx: std::ffi::c_uint) -> LLVMValue<'c, 'm> {
        let value_ref = unsafe { LLVMGetParam(self.value.value_ref, idx) };

        LLVMValue::new(value_ref)
    }
}

pub struct LLVMBasicBlock<'c> {
    pub(crate) basic_block_ref: LLVMBasicBlockRef,
    phantom_data: PhantomData<&'c LLVMContext>,
}

impl<'c> LLVMBasicBlock<'c> {
    pub fn new<'m>(context: &'c LLVMContext, function: &LLVMFunction<'c, 'm>, name: &CStr) -> Self {
        let basic_block_ref = unsafe {
            LLVMAppendBasicBlockInContext(
                context.context_ref,
                function.value.value_ref,
                name.as_ptr() as *const _,
            )
        };
        Self {
            basic_block_ref,
            phantom_data: PhantomData,
        }
    }
}
