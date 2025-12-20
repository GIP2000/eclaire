use std::{
    fs::File,
    io::{stdin, stdout, BufReader, BufWriter, Cursor, Read, Seek, SeekFrom, Stdin, Write},
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

trait ReadAndSeek: Read + Seek {}
impl<T: Read + Seek + ?Sized> ReadAndSeek for T {}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // let mut file_input = File::open("")?;
    // let file2 = &mut file_input as &mut dyn ReadAndSeek;
    // let mut test = BufReader::new(file2);
    //
    // test.seek(SeekFrom::Current(1))?;

    // let mut input_reader: BufReader<&mut dyn ReadAndSeek> =
    //     match cli.input.and_then(|x| File::open(x).ok()) {
    //         Some(f) => {
    //             file_input = f;
    //             BufReader::new(&mut file_input as &mut dyn ReadAndSeek)
    //         }
    //         None => BufReader::new(&mut stdin as &mut dyn ReadAndSeek),
    //     };
    //
    // let mut file_output;
    // let mut stdout = stdout();
    // let mut output_writer = match cli
    //     .output
    //     .and_then(|x| std::fs::OpenOptions::new().write(true).open(x).ok())
    // {
    //     Some(f) => {
    //         file_output = f;
    //         BufWriter::new(&mut file_output as &mut dyn Write)
    //     }
    //     None => BufWriter::new(&mut stdout as &mut dyn Write),
    // };
    Ok(())
}
