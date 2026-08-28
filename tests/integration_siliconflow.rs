//! Integration tests hitting the real SiliconFlow API (`api.siliconflow.cn`).
//!
//! Run with: `cargo test --features integration`
//!
//! Skipped when `SILICONFLOW_API_KEY` is unset (printed but treated as
//! success per spec §8). Configuration via environment (`.env` is loaded
//! automatically):
//!   SILICONFLOW_API_KEY      required
//!   SILICONFLOW_BASE_URL     default https://api.siliconflow.cn
//!   SILICONFLOW_TEXT_MODEL   default zai-org/GLM-5.2 (text-only)
//!   SILICONFLOW_VISION_MODEL default Qwen/Qwen3.5-397B-A17B (accepts
//!                            documents and images through `image_url`)
//!
//! Wire facts pinned by these tests (verified 2026-08 against SiliconFlow):
//! - The gateway accepts only `text` and `image_url` content parts.
//!   `file_url` and `file` parts are rejected schema-level (400 20029 /
//!   20015), including for GLM models — so the library's `Document` path
//!   must NOT be enabled on this endpoint.
//! - Document (PDF) input rides the `image_url` part with a
//!   `data:application/pdf;base64,…` URL; VL-class models parse it and read
//!   the real text (verified with a crafted marker PDF).
//! - `image_url` with base64 images works (≥28px for VL models).

#![cfg(feature = "integration")]

mod common;

use llm_adapter::{LlmError, Provider, openai::OpenAIProvider, types::*};

fn text_provider() -> Option<(OpenAIProvider, String)> {
    common::load_dotenv();
    let api_key = common::opt_env("SILICONFLOW_API_KEY")?;
    let base_url = common::opt_env("SILICONFLOW_BASE_URL")
        .unwrap_or_else(|| "https://api.siliconflow.cn".into());
    let model =
        common::opt_env("SILICONFLOW_TEXT_MODEL").unwrap_or_else(|| "zai-org/GLM-5.2".into());
    Some((
        OpenAIProvider::builder(api_key.clone())
            .base_url(base_url)
            .build(),
        model,
    ))
}

fn vision_provider() -> Option<(OpenAIProvider, String)> {
    common::load_dotenv();
    let api_key = common::opt_env("SILICONFLOW_API_KEY")?;
    let base_url = common::opt_env("SILICONFLOW_BASE_URL")
        .unwrap_or_else(|| "https://api.siliconflow.cn".into());
    let model = common::opt_env("SILICONFLOW_VISION_MODEL")
        .unwrap_or_else(|| "Qwen/Qwen3.5-397B-A17B".into());
    Some((
        OpenAIProvider::builder(api_key.clone())
            .base_url(base_url)
            .build(),
        model,
    ))
}

#[tokio::test]
async fn integration_simple_chat() {
    let Some((p, model)) = text_provider() else {
        println!("[skip] SILICONFLOW_API_KEY not set");
        return;
    };
    let req = ChatRequest::builder(model, 256)
        .message(Message::User(vec![RequestContent::Text(
            "Reply with exactly: OK".into(),
        )]))
        .build();
    let resp = p.chat(&req).await.expect("chat failed");
    assert!(!resp.text().is_empty());
    assert!(matches!(
        resp.stop_reason,
        StopReason::EndTurn | StopReason::MaxTokens
    ));
}

/// 64x64 solid red PNG — big enough for VL models' 28px minimum.
fn red_png_base64() -> String {
    use std::io::Write as _;
    let w: u32 = 64;
    let h: u32 = 64;
    let mut raw = Vec::with_capacity((h * (1 + w * 3)) as usize);
    for _ in 0..h {
        raw.push(0u8);
        for _ in 0..w {
            raw.extend_from_slice(&[255, 0, 0]);
        }
    }
    let mut png = Vec::new();
    png.extend_from_slice(&[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]);
    let mut chunk = |t: [u8; 4], d: &[u8], out: &mut Vec<u8>| {
        out.extend_from_slice(&(d.len() as u32).to_be_bytes());
        out.extend_from_slice(&t);
        out.extend_from_slice(d);
        let mut crc = flate2_crc32(&t, d);
        out.extend_from_slice(&crc.to_be_bytes());
    };
    fn flate2_crc32(t: &[u8; 4], d: &[u8]) -> u32 {
        let mut table = [0u32; 256];
        for i in 0..256u32 {
            let mut c = i;
            for _ in 0..8 {
                c = if c & 1 != 0 {
                    0xEDB8_8320 ^ (c >> 1)
                } else {
                    c >> 1
                };
            }
            table[i as usize] = c;
        }
        let mut crc = 0xFFFF_FFFFu32;
        for b in t.iter().chain(d.iter()) {
            crc = table[((crc ^ *b as u32) & 0xFF) as usize] ^ (crc >> 8);
        }
        crc ^ 0xFFFF_FFFF
    }
    let mut ihdr = Vec::new();
    write!(&mut ihdr, "").unwrap();
    ihdr.clear();
    ihdr.extend_from_slice(&w.to_be_bytes());
    ihdr.extend_from_slice(&h.to_be_bytes());
    ihdr.extend_from_slice(&[8, 2, 0, 0, 0]);
    chunk(*b"IHDR", &ihdr, &mut png);
    // zlib stream: 0x78 0x01 header + deflate of raw + adler32
    let compressed = zlib_compress(&raw);
    chunk(*b"IDAT", &compressed, &mut png);
    chunk(*b"IEND", &[], &mut png);
    base64_encode(&png)
}

fn zlib_compress(data: &[u8]) -> Vec<u8> {
    use std::io::Read as _;
    // Stored (uncompressed) deflate blocks: simple and dependency-free.
    let mut out = vec![0x78, 0x01];
    let mut chunks = data.chunks(65535).peekable();
    if chunks.peek().is_none() {
        out.extend_from_slice(&[0x01, 0x00, 0x00, 0xFF, 0xFF]);
    }
    while let Some(c) = chunks.next() {
        let last = chunks.peek().is_none();
        out.push(if last { 0x01 } else { 0x00 });
        let len = c.len() as u16;
        out.extend_from_slice(&len.to_le_bytes());
        out.extend_from_slice(&(!len).to_le_bytes());
        out.extend_from_slice(c);
    }
    let mut a: u32 = 1;
    let mut b: u32 = 0;
    for &byte in data {
        a = (a + byte as u32) % 65521;
        b = (b + a) % 65521;
    }
    out.extend_from_slice(&((b << 16) | a).to_be_bytes());
    out
}

fn base64_encode(data: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for c in data.chunks(3) {
        let b = [c[0], *c.get(1).unwrap_or(&0), *c.get(2).unwrap_or(&0)];
        let n = ((b[0] as u32) << 16) | ((b[1] as u32) << 8) | b[2] as u32;
        out.push(T[(n >> 18) as usize & 63] as char);
        out.push(T[(n >> 12) as usize & 63] as char);
        out.push(if c.len() > 1 {
            T[(n >> 6) as usize & 63] as char
        } else {
            '='
        });
        out.push(if c.len() > 2 {
            T[n as usize & 63] as char
        } else {
            '='
        });
    }
    out
}

/// Minimal single-page PDF embedding a distinctive marker sentence.
fn marker_pdf_base64() -> String {
    let marker = "SILICONFLOW_MARKER_7391_UNIQUE";
    let text = format!("BT /F1 18 Tf 20 50 Td ({marker}) Tj ET");
    let pdf = format!(
        "%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n\
         2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n\
         3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 300 100]/Contents 4 0 R\
         /Resources<</Font<</F1 5 0 R>>>>>>endobj\n\
         4 0 obj<</Length {len}>>stream\n{text}\nendstream\nendobj\n\
         5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj\n\
         trailer<</Root 1 0 R>>\n%%EOF",
        len = text.len()
    );
    base64_encode(pdf.as_bytes())
}

#[tokio::test]
async fn integration_image_input_base64() {
    let Some((p, model)) = vision_provider() else {
        println!("[skip] SILICONFLOW_API_KEY not set");
        return;
    };
    let req = ChatRequest::builder(model, 256)
        .message(Message::User(vec![
            RequestContent::Text("What color is this image? One word.".into()),
            RequestContent::Image(ImageSource::Base64 {
                media_type: "image/png".into(),
                data: red_png_base64(),
            }),
        ]))
        .build();
    let resp = p.chat(&req).await.expect("image chat failed");
    assert!(
        resp.text().to_lowercase().contains("red"),
        "expected 'red' in model reply, got: {}",
        resp.text()
    );
}

#[tokio::test]
async fn integration_document_via_image_url_channel() {
    // Wire fact: SiliconFlow has NO file/file_url part; PDFs ride image_url as
    // a data:application/pdf URL. The library's ImageSource::Base64 renders
    // exactly that shape — so a PDF "just works" through the image path on
    // VL-class models.
    let Some((p, model)) = vision_provider() else {
        println!("[skip] SILICONFLOW_API_KEY not set");
        return;
    };
    let req = ChatRequest::builder(model, 512)
        .message(Message::User(vec![
            RequestContent::Text(
                "What exact marker text does this document contain? Quote it.".into(),
            ),
            RequestContent::Image(ImageSource::Base64 {
                media_type: "application/pdf".into(),
                data: marker_pdf_base64(),
            }),
        ]))
        .build();
    let resp = p.chat(&req).await.expect("pdf-via-image chat failed");
    assert!(
        resp.text().contains("SILICONFLOW_MARKER_7391"),
        "model must read the real marker from the PDF, got: {}",
        resp.text()
    );
}

#[tokio::test]
async fn integration_document_capability_rejected_pre_dispatch() {
    // Wire fact: SiliconFlow rejects file_url/file parts schema-level (400
    // 20029). The library must therefore keep `documents` off by default on
    // this endpoint — Document content fails at construction time, no HTTP.
    let Some((p, _)) = text_provider() else {
        println!("[skip] SILICONFLOW_API_KEY not set");
        return;
    };
    assert!(!p.capabilities().supports_documents());
    let req = ChatRequest::builder("Qwen/Qwen3.5-397B-A17B", 256)
        .message(Message::User(vec![RequestContent::Document(
            DocumentSource::Url {
                url: "https://example.com/x.pdf".into(),
            },
        )]))
        .build();
    let err = p.chat(&req).await.unwrap_err();
    match err {
        LlmError::InvalidRequest(msg) => assert!(
            msg.contains("documents capability disabled"),
            "expected capability error, got: {msg}"
        ),
        other => panic!("expected InvalidRequest, got {other:?}"),
    }
}
