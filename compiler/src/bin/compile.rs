use clap::Parser;
use eclaire::{all, fatal, info, parser::parse, trace, utils::logger::Level};
use std::{
    fs::File,
    io::{stdin, Read},
    path::PathBuf,
};

#[derive(Parser, Debug)]
struct Cli {
    /// Optional file path to use as input
    /// defaults to stdin
    input: Option<PathBuf>,

    /// Optional Debug infromation
    #[arg(
        short,
        long,
        action = clap::ArgAction::SetTrue,
        value_parser = |x: &str| -> anyhow::Result<Level> {Ok(if x == "true" {Level::debug()} else {Level::none()})})
    ]
    debug: Level,

    /// Optional trace infromation
    #[arg(
        short,
        long,
        action = clap::ArgAction::SetTrue,
        value_parser = |x: &str| -> anyhow::Result<Level> {Ok(if x == "true" {Level::trace()} else {Level::none()})})
    ]
    trace: Level,

    /// Optional info infromation
    #[arg(
        short,
        long,
        action = clap::ArgAction::SetTrue,
        value_parser = |x: &str| -> anyhow::Result<Level> {Ok(if x == "true" {Level::info()} else {Level::none()})})
    ]
    info: Level,

    /// Optional warning infromation
    #[arg(
        short,
        long,
        action = clap::ArgAction::SetTrue,
        value_parser = |x: &str| -> anyhow::Result<Level> {Ok(if x == "true" {Level::warn()} else {Level::none()})})
    ]
    warn: Level,

    /// Optional Output File
    /// deafults to stdout
    #[arg(short, long)]
    output: Option<PathBuf>,
}

impl Cli {
    fn get_log_level(&self) -> u8 {
        self.debug
            | self.trace
            | self.info
            | self.warn
            | Level::error()
            | Level::fatal()
            | Level::all()
    }
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    Level::set_log_level(cli.get_log_level());

    let mut source_code = String::new();
    let mut file: std::fs::File;
    let mut stdin = stdin();

    trace!("started reading source code");

    let reader: &mut dyn Read = match cli.input.as_ref() {
        Some(path) => {
            file = File::open(path).map_err(|err| {
                fatal!("Failed to open input file {:?}", err);
                err
            })?;
            &mut file as &mut dyn Read
        }
        None => &mut stdin as &mut dyn Read,
    };

    reader.read_to_string(&mut source_code).map_err(|err| {
        match cli.input.as_ref() {
            Some(path) => {
                fatal!("Failed to read input file: {:?}: {:?}", path, err);
            }
            None => fatal!("Failed to read from stdin: {:?}", err),
        }
        err
    })?;

    let source_code = source_code.into_boxed_str();
    trace!("Finished reading source code");
    info!("source code = {}", source_code);

    match parse(&source_code) {
        Ok(x) => {
            all!("Finished succesfully");
            info!("AST: {x:?}");
            Ok(())
        }
        Err(x) => {
            fatal!("Error making program {:?}", x);
            Err(anyhow::anyhow!("Error couldn't make program: {:?}", x))
        }
    }
}
