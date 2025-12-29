use std::{
    fs::File,
    io::{stdin, Read},
    path::PathBuf,
};

use eclaire::parser::parse;

use clap::Parser;

#[derive(Parser, Debug)]
struct Cli {
    /// Optional file path to use as input
    /// defaults to stdin
    input: Option<PathBuf>,

    /// Optional Debug infromation
    #[arg(short, long, action = clap::ArgAction::SetTrue)]
    debug: bool,

    /// Optional Output File
    /// deafults to stdout
    #[arg(short, long)]
    output: Option<PathBuf>,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let mut source_code = String::new();
    let mut file: std::fs::File;
    let mut stdin = stdin();

    let reader: &mut dyn Read = match cli.input {
        Some(path) => {
            file = File::open(path)?;
            &mut file as &mut dyn Read
        }
        None => &mut stdin as &mut dyn Read,
    };
    reader.read_to_string(&mut source_code)?;

    let source_code = source_code.into_boxed_str();

    parse(&source_code)?;

    Ok(())
}
