use anyhow::{bail, Result};
use std::{ffi::CStr, marker::PhantomData};

use llvm_sys::{
    execution_engine::{
        LLVMCreateExecutionEngineForModule, LLVMExecutionEngineRef, LLVMGetFunctionAddress,
        LLVMLinkInMCJIT,
    },
    target::{LLVM_InitializeNativeAsmPrinter, LLVM_InitializeNativeTarget},
};

use crate::ir::llvm_wrapper::LLVMModule;

pub struct ExecutionEngine<'m> {
    execution_engine_ref: LLVMExecutionEngineRef,
    phantom_data: PhantomData<&'m LLVMModule>,
}

impl<'m> ExecutionEngine<'m> {
    pub fn new(module: &'m LLVMModule) -> Result<Self> {
        unsafe {
            LLVMLinkInMCJIT();
            LLVM_InitializeNativeTarget();
            LLVM_InitializeNativeAsmPrinter();
        };

        let mut ee = std::mem::MaybeUninit::uninit();
        let mut err = unsafe { std::mem::zeroed() };

        let execution_engine_ref = if unsafe {
            LLVMCreateExecutionEngineForModule(ee.as_mut_ptr(), module.module_ref, &mut err) != 0
        } {
            if err.is_null() {
                bail!("Failed to create engine: error not parseable");
            } else {
                let err_msg = unsafe { CStr::from_ptr(err) };
                bail!("Failed to create engine: {:?}", err_msg);
            }
        } else {
            unsafe { ee.assume_init() }
        };

        Ok(Self {
            execution_engine_ref,
            phantom_data: PhantomData,
        })
    }

    pub fn get_function_ref(&self, name: &CStr) -> u64 {
        unsafe { LLVMGetFunctionAddress(self.execution_engine_ref, name.as_ptr() as *const _) }
    }
}
