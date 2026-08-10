use llvm_sys::{
    orc2::lljit::{
        LLVMOrcCreateLLJIT, LLVMOrcCreateLLJITBuilder, LLVMOrcLLJITBuilderRef, LLVMOrcLLJITRef,
    },
    target::{LLVM_InitializeNativeAsmParser, LLVM_InitializeNativeTarget},
};

// use crate::ir::llvm_wrapper::{LLVMContext, LLVMModule};

pub struct LLVMJitCompiler {
    // context: LLVMContext,
    // module: LLVMModule,
    pub jit_builder: LLVMOrcLLJITBuilderRef,
    pub orc: LLVMOrcLLJITRef,
}

impl LLVMJitCompiler {
    pub fn new() -> Self {
        let (jit_builder, orc, error) = unsafe {
            LLVM_InitializeNativeTarget();
            LLVM_InitializeNativeAsmParser();

            let jit_builder = LLVMOrcCreateLLJITBuilder();
            let mut orc = std::ptr::null_mut();

            let error = LLVMOrcCreateLLJIT(&mut orc, jit_builder);

            (jit_builder, orc, error)
        };

        if !error.is_null() {
            panic!("There is an error initalizing the JIT");
        }

        // let context = LLVMContext::new();
        // let module = LLVMModule::new();

        Self {
            // context,
            // module,
            jit_builder,
            orc,
        }
    }
}
