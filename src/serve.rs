use serde::{Deserialize, Serialize};
use tiny_http::{Header, Method, Response, Server, StatusCode};

use crate::config::BLOCK_SIZE;
use crate::model::GPTModel;
use crate::rng::Rng;
use crate::tokenizer::Tokenizer;
use crate::train::generate_cpu;

#[derive(Deserialize)]
struct InferRequest {
    prompt: String,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default = "default_temperature")]
    temperature: f32,
}

fn default_max_tokens() -> usize { unsafe { BLOCK_SIZE } }
fn default_temperature() -> f32 { 0.7 }

#[derive(Serialize)]
struct Usage {
    prompt_tokens: usize,
    completion_tokens: usize,
}

#[derive(Serialize)]
struct InferResponse {
    text: String,
    model: String,
    usage: Usage,
}

fn json_content_type() -> Header {
    Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).unwrap()
}

fn error_response(msg: &str, code: u16) -> Response<std::io::Cursor<Vec<u8>>> {
    let body = serde_json::json!({ "error": msg }).to_string();
    Response::from_string(body)
        .with_status_code(StatusCode(code))
        .with_header(json_content_type())
}

pub fn run_server(
    addr: &str,
    model: &GPTModel,
    tokenizer: &Tokenizer,
    model_name: &str,
    api_key: Option<&str>,
) {
    let server = Server::http(addr).unwrap_or_else(|e| {
        eprintln!("Failed to bind to {}: {}", addr, e);
        std::process::exit(1);
    });

    println!("Server listening on http://{}", addr);
    println!("POST http://{}/ with JSON body:", addr);
    println!("  {{\"prompt\": \"Once upon a time\", \"max_tokens\": 200, \"temperature\": 0.8}}");

    println!();

    for mut request in server.incoming_requests() {
        if *request.method() != Method::Post {
            let _ = request.respond(error_response("Method Not Allowed", 405));
            continue;
        }

        // Optional bearer token auth
        if let Some(key) = api_key {
            let expected = format!("Bearer {}", key);
            let authed = request
                .headers()
                .iter()
                .any(|h| h.field.equiv("Authorization") && h.value.as_str() == expected);
            if !authed {
                let _ = request.respond(error_response("Unauthorized", 401));
                continue;
            }
        }

        // Read body
        let mut body = String::new();
        if request.as_reader().read_to_string(&mut body).is_err() {
            let _ = request.respond(error_response("Failed to read request body", 400));
            continue;
        }

        // Parse request JSON
        let req: InferRequest = match serde_json::from_str(&body) {
            Ok(r) => r,
            Err(e) => {
                let _ = request.respond(error_response(&e.to_string(), 400));
                continue;
            }
        };

        let prompt_tokens = tokenizer.encode(&req.prompt).len();

        // Seed RNG from subsecond timestamp for variety per request
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.subsec_nanos() as u64)
            .unwrap_or(42);
        let mut rng = Rng::new(seed);

        let max_tokens = req.max_tokens;
        eprintln!("[serve] prompt={:?} max_tokens={} temperature={}", &req.prompt, max_tokens, req.temperature);

        let completion = generate_cpu(
            model,
            tokenizer,
            &req.prompt,
            max_tokens,
            req.temperature,
            0.9,
            &mut rng,
        );

        let completion_tokens = tokenizer.encode(&completion).len();

        let resp = InferResponse {
            text: completion,
            model: model_name.to_string(),
            usage: Usage { prompt_tokens, completion_tokens },
        };

        let json = serde_json::to_string(&resp)
            .unwrap_or_else(|_| "{\"text\":\"\"}".to_string());

        let _ = request.respond(
            Response::from_string(json).with_header(json_content_type()),
        );
    }
}
