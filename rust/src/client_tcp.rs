// Uso: client_tcp <output_folder> <ip:porta> [<ip:porta> ...]
//
// O handshake HTTP (request_handle_tcp / start_scanoutput) é feito pelo
// ScanManager em Python. Este binário apenas conecta nas portas TCP já
// abertas e grava os dados brutos em <output_folder>/<ip>.bin.

use std::env;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use tokio::io::AsyncReadExt;
use tokio::net::TcpStream;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 3 {
        eprintln!("Uso: {} <output_folder> <ip:porta> [<ip:porta> ...]", args[0]);
        eprintln!("Exemplo: {} ./pointcloud/scan/ 192.168.1.11:53509 192.168.1.12:42275", args[0]);
        std::process::exit(1);
    }

    let output_folder = &args[1];
    let addresses = &args[2..];

    let mut tasks = Vec::new();

    for addr in addresses {
        let addr = addr.clone();
        let folder = output_folder.clone();

        let task = tokio::spawn(async move {
            // Extrai o IP da string "ip:porta" para usar como nome de arquivo
            let ip = addr.split(':').next().unwrap_or(&addr);
            let file_path = Path::new(&folder).join(format!("{}.bin", ip));

            let file = match File::create(&file_path) {
                Ok(f) => f,
                Err(e) => {
                    eprintln!("[{}] Erro ao criar arquivo {:?}: {}", addr, file_path, e);
                    return;
                }
            };
            let mut writer = BufWriter::new(file);

            println!("[{}] Conectando...", addr);
            let mut stream = match TcpStream::connect(&addr).await {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("[{}] Erro ao conectar: {}", addr, e);
                    return;
                }
            };
            println!("[{}] Conectado. Gravando em {:?}", addr, file_path);

            let mut buffer = [0u8; 8192];
            loop {
                match stream.read(&mut buffer).await {
                    Ok(0) => {
                        println!("[{}] Conexão encerrada pelo sensor.", addr);
                        break;
                    }
                    Ok(n) => {
                        if let Err(e) = writer.write_all(&buffer[..n]) {
                            eprintln!("[{}] Erro ao gravar: {}", addr, e);
                            break;
                        }
                    }
                    Err(e) => {
                        eprintln!("[{}] Erro de leitura TCP: {}", addr, e);
                        break;
                    }
                }
            }

            if let Err(e) = writer.flush() {
                eprintln!("[{}] Erro ao fechar arquivo: {}", addr, e);
            }
        });

        tasks.push(task);
    }

    for task in tasks {
        let _ = task.await;
    }

    Ok(())
}
