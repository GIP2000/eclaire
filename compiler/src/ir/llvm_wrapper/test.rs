use crate::ir::llvm_wrapper::{ir_components::LLVMFunctionType, LLVMContext, LLVMModule};
use llvm_sys::core::LLVMInt32TypeInContext;

#[test]
fn test_ir() {
    let ctx = LLVMContext::new();
    let builder = ctx.make_builder();

    let module = LLVMModule::new();

    let i32t = ctx.make_type(LLVMInt32TypeInContext);
    let mut args = [i32t, i32t, i32t];
    let function_type = LLVMFunctionType::new(i32t, &mut args);
    let function = module.make_function_val(c"sum", function_type);

    let bb = ctx.make_bb(&function, c"entry");

    builder.move_to_end(&bb);

    let x = function.get_arg(0);
    let y = function.get_arg(1);
    let z = function.get_arg(2);

    let sum = builder.add(x, y, c"sum.1");
    let sum = builder.add(sum, z, c"sum.2");
    builder.ret(sum);

    module.dump();

    let engine = module.make_execution_engine();
    assert!(engine.is_ok());
    let engine = engine.unwrap();

    let llvm_sum_func: unsafe extern "C" fn(x: i32, y: i32, z: i32) -> i32 =
        unsafe { std::mem::transmute(engine.get_function_ref(c"sum")) };

    let llvm_result = unsafe { llvm_sum_func(1, 2, 3) };

    assert_eq!(llvm_result, 6);
}
