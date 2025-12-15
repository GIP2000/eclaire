use std::{
    fs::File,
    io::{stdin, stdout, BufReader, BufWriter, Write},
    path::PathBuf,
};

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

fn main() {
    let cli = Cli::parse();

    let mut file_input;
    let mut stdin = stdin();
    let mut input_reader = match cli.input.and_then(|x| File::open(x).ok()) {
        Some(f) => {
            file_input = f;
            BufReader::new(&mut file_input as &mut dyn std::io::Read)
        }
        None => BufReader::new(&mut stdin as &mut dyn std::io::Read),
    };

    let mut file_output;
    let mut stdout = stdout();
    let mut output_writer = match cli
        .output
        .and_then(|x| std::fs::OpenOptions::new().write(true).open(x).ok())
    {
        Some(f) => {
            file_output = f;
            BufWriter::new(&mut file_output as &mut dyn Write)
        }
        None => BufWriter::new(&mut stdout as &mut dyn Write),
    };
}
