use reqwest::Client;
use serde::Deserialize;
use std::env;
use std::time::Duration;
use tokio::io::AsyncReadExt;
use tokio::net::TcpStream;
use tokio::time::sleep;
use std::fs::File;
use std::io::{BufWriter, Write}; 

// Estruturas para converter o JSON de resposta do sensor em tipos do Rust
#[derive(Deserialize, Debug)]
struct HandleResponse {
    error_code: i32,
    handle: String,
    port: u16,
}

#[derive(Deserialize, Debug)]
struct GenericResponse {
    error_code: i32,
    error_text: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();

    // args[0] é o nome do próprio programa. args[1] será o nosso IP.
    if args.len() < 2 {
        eprintln!("Erro: IP do sensor não fornecido.");
        eprintln!("Uso correto: {} <IP_DO_SENSOR>", args[0]);
        eprintln!("Exemplo:     {} 192.168.0.100", args[0]);
        std::process::exit(1); // Encerra o programa com código de erro
    }

    // Substitua pelo IP real do seu LiDAR R2000
    let sensor_ip = &args[1];

    let file_name = format!("sensor_{}.bin", sensor_ip.replace(".", "_"));
    let file = File::create(&file_name)?;

    let mut writer = BufWriter::new(file);

    let client = Client::new();

    // ====================================================================
    // PASSO 1: Solicitar o Handle e a Porta TCP
    // ====================================================================
    println!("1. Solicitando handle...");
    let handle_url = format!("http://{}/cmd/request_handle_tcp?packet_type=A", sensor_ip);
    
    let resp = client.get(&handle_url)
        .send().await?
        .json::<HandleResponse>().await?;

    if resp.error_code != 0 {
        eprintln!("Erro ao solicitar handle. O sensor pode estar ocupado.");
        return Ok(());
    }

    let handle = resp.handle;
    let tcp_port = resp.port;
    println!("   -> Handle obtido: {}", handle);
    println!("   -> Porta TCP liberada: {}", tcp_port);

    // ====================================================================
    // PASSO 2: Iniciar a medição (Start Scan)
    // ====================================================================
    println!("2. Iniciando envio de dados do scan...");
    let start_url = format!("http://{}/cmd/start_scanoutput?handle={}", sensor_ip, handle);
    
    let start_resp = client.get(&start_url)
        .send().await?
        .json::<GenericResponse>().await?;

    if start_resp.error_code != 0 {
        eprintln!("   -> Erro ao iniciar scan: {}", start_resp.error_text);
        return Ok(());
    }
    println!("   -> Scan iniciado com sucesso!");

    // ====================================================================
    // PASSO 3: Tarefa em Background para alimentar o Watchdog
    // ====================================================================
    // Clonamos variáveis para mover para dentro da thread assíncrona
    let watchdog_client = client.clone();
    let watchdog_handle = handle.clone();
    let watchdog_ip = sensor_ip.to_string();

    tokio::spawn(async move {
        let watchdog_url = format!("http://{}/cmd/feed_watchdog?handle={}", watchdog_ip, watchdog_handle);
        
        loop {
            // Dorme por 15 segundos (o timeout do sensor geralmente é 60s)
            sleep(Duration::from_secs(15)).await; 
            
            match watchdog_client.get(&watchdog_url).send().await {
                Ok(_) => println!("[Watchdog] Ping enviado com sucesso."),
                Err(e) => eprintln!("[Watchdog] Falha de comunicação: {}", e),
            }
        }
    });

    // ====================================================================
    // PASSO 4: Conectar ao Stream TCP e ler os dados brutos
    // ====================================================================
    println!("3. Conectando ao stream TCP na porta {}...", tcp_port);
    let address = format!("{}:{}", sensor_ip, tcp_port);
    let mut stream = TcpStream::connect(address).await?;
    println!("   -> Conectado! Lendo nuvem de pontos...\n");

    let mut buffer = [0u8; 8192]; // Buffer razoável para os pacotes do scanner

    loop {
        match stream.read(&mut buffer).await {
            Ok(0) => {
                println!("A conexão TCP foi encerrada pelo sensor.");
                break;
            }
            Ok(n) => {
                // Aqui os dados brutos (PFSDP) chegam. 
                // O próximo passo seria analisar o cabeçalho binário.
                writer.write_all(&buffer[..n])?;
                println!("Recebidos {} bytes de dados...", n);
                
                // Exemplo: Imprimir os primeiros 4 bytes (Magic Bytes do protocolo)
                // println!("Cabeçalho: {:02X?}", &buffer[0..4]);
            }
            Err(e) => {
                eprintln!("Erro crítico ao ler da porta TCP: {}", e);
                break;
            }
        }
    }

    Ok(())
}
