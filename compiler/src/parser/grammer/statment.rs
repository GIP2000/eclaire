use lexer::LexerIterator;

use crate::{
    lexer::{LexToken, MyLexerError},
    parser::{
        grammer::{
            assignment::{Assignment, AssignmentType},
            expression::{Expression, TypeResp},
        },
        symbol_table::{
            CompareTypes, DeclNode, SymbolTableDecl, SymbolTableError, SymbolTableType,
        },
        Parse, ParserInto, Result,
    },
    trace,
};

#[derive(Debug, Clone)]
pub enum Statment {
    Assignment(Assignment),
    Expression(Expression),
    Return(bool, Expression),
}
impl Statment {
    pub fn type_check(
        &self,
        type_defs: (&SymbolTableType, usize),
        decls: &mut SymbolTableDecl,
        func_ret_type: &TypeResp,
        block_ret_type: &TypeResp,
    ) -> std::result::Result<(), SymbolTableError> {
        Ok(match self {
            Statment::Assignment(assignment) => match assignment.assignment_type {
                AssignmentType::Let(is_mut) => {
                    match (assignment.expr.as_ref(), assignment.data_type.as_ref()) {
                        (Some(assignment_expr), None) => {
                            let typeresp =
                                assignment_expr.get_type(type_defs, decls, func_ret_type)?;

                            decls.insert_from_resp(
                                assignment.ident.clone(),
                                typeresp,
                                is_mut,
                                type_defs.0,
                            )?;
                        }
                        (None, Some(type_name)) => {
                            decls.insert(
                                assignment.ident.clone(),
                                DeclNode::new(type_name.clone(), is_mut),
                            )?;
                        }
                        (Some(assignment_expr), Some(type_name))
                            if {
                                assignment_expr
                                    .get_type(type_defs, decls, func_ret_type)?
                                    .are_types_eq(type_name, type_defs)

                                // let type_resp =
                                //     assignment_expr.get_type(type_defs, decls, func_ret_type)?;
                                //
                                // match &type_resp.get_root_type() {
                                //     TypeResp::Pointer(_, _) => unreachable!("I got the root type"),
                                //     x @ TypeResp::IdentRef(_) => {
                                //         x.are_types_eq(type_name, type_defs)
                                //     }
                                //     TypeResp::Void => {
                                //         unimplemented!("figure out how to handle void type")
                                //     }
                                //     TypeResp::IntLike => {
                                //         let type_data = type_defs.get_until_root(type_name).ok_or(
                                //             SymbolTableError::TypeError(
                                //                 format!("Type `{}` not found", type_name).into(),
                                //             ),
                                //         )?;
                                //
                                //         matches!(
                                //             type_data.type_info,
                                //             TypeDefInfoType::TypeDefPrim(PrimativeType {
                                //                 size: _,
                                //                 like: PrimativeLike::UInt | PrimativeLike::SInt,
                                //                 is_default: _
                                //             })
                                //         )
                                //     }
                                //     TypeResp::FloatLike => {
                                //         let type_data = type_defs.get_until_root(type_name).ok_or(
                                //             SymbolTableError::TypeError(
                                //                 format!("Type `{}` not found", type_name).into(),
                                //             ),
                                //         )?;
                                //
                                //         matches!(
                                //             type_data.type_info,
                                //             TypeDefInfoType::TypeDefPrim(PrimativeType {
                                //                 size: _,
                                //                 like: PrimativeLike::Float,
                                //                 is_default: _
                                //             })
                                //         )
                                //     }
                                //     TypeResp::CharLike => {
                                //         let type_data = type_defs.get_until_root(type_name).ok_or(
                                //             SymbolTableError::TypeError(
                                //                 format!("Type `{}` not found", type_name).into(),
                                //             ),
                                //         )?;
                                //
                                //         matches!(
                                //             type_data.type_info,
                                //             TypeDefInfoType::TypeDefPrim(PrimativeType {
                                //                 size: _,
                                //                 like: PrimativeLike::Char,
                                //                 is_default: _
                                //             })
                                //         )
                                //     }
                                //     TypeResp::BoolLike => {
                                //         let type_data = type_defs.get_until_root(type_name).ok_or(
                                //             SymbolTableError::TypeError(
                                //                 format!("Type `{}` not found", type_name).into(),
                                //             ),
                                //         )?;
                                //
                                //         matches!(
                                //             type_data.type_info,
                                //             TypeDefInfoType::TypeDefPrim(PrimativeType {
                                //                 size: _,
                                //                 like: PrimativeLike::Bool,
                                //                 is_default: _
                                //             })
                                //         )
                                //     }
                                // }
                            } =>
                        {
                            decls.insert(
                                assignment.ident.clone(),
                                DeclNode::new(type_name.clone(), is_mut),
                            )?;
                        }
                        // TODO: put a better error
                        (Some(_), Some(_)) | (None, None) => {
                            return Err(SymbolTableError::TypeError(
                                "Not enough info to make an infrence".into(),
                            ));
                        }
                    }
                }
                AssignmentType::Const => {}
            },
            Statment::Expression(expr) => {
                expr.get_type(type_defs, decls, func_ret_type)?;
            }
            Statment::Return(is_func, expr) => {
                let r_type = expr.get_type(type_defs, decls, func_ret_type)?;
                r_type
                    .are_types_eq(
                        is_func.then_some(func_ret_type).unwrap_or(&block_ret_type),
                        type_defs,
                    )
                    .then_some(())
                    .ok_or(SymbolTableError::TypeError(
                        "Type Mismatch: return doesn't expected type".into(),
                    ))?;
            }
        })
    }
}

impl Parse for Statment {
    fn from_lexer<'a>(
        token_stream: &mut impl LexerIterator<'a, LexToken<'a>, MyLexerError>,
        symbol_table: &mut SymbolTableType,
    ) -> Result<Self> {
        trace!("Entering Statment");

        token_stream
            .parse(symbol_table)
            .map(|x| Self::Assignment(x))
            .or_else(|_| {
                token_stream.parse(symbol_table).map(|x| {
                    match token_stream.next_matches(LexToken::SemiColon) {
                        Ok(_) => Self::Expression(x),
                        Err(_) => Self::Return(false, x),
                    }
                })
            })
            .or_else(|_| {
                token_stream.next_matches(LexToken::Return)?;
                let expr: Expression = token_stream.parse(symbol_table)?;
                Ok(Self::Return(true, expr))
            })
    }
}
