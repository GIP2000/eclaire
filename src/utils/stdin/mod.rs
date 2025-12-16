use std::io::{Read, Seek, SeekFrom, Stdin};

struct SeekableStdin {
    inner: Stdin,
    buf: Vec<u8>,
    cursor: usize,
}

impl Seek for SeekableStdin {
    fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
        use SeekFrom::*;
        match pos {
            Start(dx) => dx,
            End(dx) => todo!(),
            Current(dx) => todo!(),
        }
    }
}

impl Read for SeekableStdin {
    fn read(&mut self, mut buf: &mut [u8]) -> std::io::Result<usize> {
        let to_read = buf.len();

        if let Some(new_to_read) = (self.cursor + to_read).checked_sub(self.buf.len()) {
            let mut inner_buf = vec![0u8; new_to_read];
            self.inner.read(&mut inner_buf)?;
            self.buf.append(&mut inner_buf);
        }

        let written =
            buf.write(&self.buf[self.cursor..(self.cursor + to_read).min(self.buf.len())])?;
        self.cursor += written;
        return Ok(written);
    }
}
