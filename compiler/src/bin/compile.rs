use clap::Parser;
use eclaire::{all, fatal, info, parser::parse, trace, utils::logger::Level};
use std::{
    fs::File,
    io::{stdin, stdout, Read},
    path::PathBuf,
};

macro_rules! parse_level {
    ($val: expr) => {
        |x: &str| -> anyhow::Result<eclaire::utils::logger::Level> {
            Ok((x == "true")
                .then_some($val)
                .unwrap_or(eclaire::utils::logger::Level::none()))
        }
    };
}

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
        value_parser = parse_level!(Level::debug())
    )]
    debug: Level,

    /// Optional trace infromation
    #[arg(
        short,
        long,
        action = clap::ArgAction::SetTrue,
        value_parser = parse_level!(Level::trace())

    )]
    trace: Level,

    /// Optional info infromation
    #[arg(
        short,
        long,
        action = clap::ArgAction::SetTrue,
        value_parser = parse_level!(Level::info())
    )]
    info: Level,

    /// Optional warning infromation
    #[arg(
        short,
        long,
        action = clap::ArgAction::SetTrue,
        value_parser = parse_level!(Level::warn())
    )]
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

    let table = match parse(&source_code) {
        Ok(table) => {
            all!("Finished succesfully");
            info!("symbol table: {table:?}");
            table
        }
        Err(x) => {
            fatal!("Error making program {}", x);
            return Err(anyhow::anyhow!("Error couldn't make program: {:?}", x));
        }
    };

    table.type_check().map_err(|err| {
        fatal!("Error making program: {}", err);
        err
    })?;

    let mut file: std::fs::File;
    let mut stdout = stdout();
    let writer = match cli.output.as_ref() {
        Some(path) => {
            file = std::fs::OpenOptions::new()
                .write(true)
                .create(true)
                .open(path)
                .map_err(|err| {
                    fatal!("Failed to open file {:?}: {:?}", path, err);
                    err
                })?;
            &mut file as &mut dyn std::io::Write
        }
        None => &mut stdout as &mut dyn std::io::Write,
    };

    writeln!(writer, "{:?}", table).map_err(|err| {
        match cli.output.as_ref() {
            Some(path) => fatal!("Failed to write to file {:?}: {:?}", path, err),
            None => fatal!("Failed to write to stdout: {:?}", err),
        }
        err
    })?;

    Ok(())
}
