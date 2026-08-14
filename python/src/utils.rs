use pyo3::prelude::*;
use pyo3::types::PyTuple;
use serde_json::json;

pub fn infer_json_schema_type(annotation: &Bound<'_, PyAny>) -> serde_json::Value {
    if let (Ok(origin), Ok(args)) = (
        annotation.getattr("__origin__"),
        annotation.getattr("__args__"),
    ) {
        let origin_name: String = origin
            .getattr("__name__")
            .and_then(|n| n.extract())
            .unwrap_or_default();

        let args_vec: Vec<Bound<'_, PyAny>> = args
            .downcast_into::<PyTuple>()
            .map(|t| t.iter().collect())
            .unwrap_or_default();

        let is_none_type = |a: &Bound<'_, PyAny>| -> bool {
            a.getattr("__name__")
                .and_then(|n| n.extract::<String>())
                .unwrap_or_default()
                == "NoneType"
        };
        
        let mut has_none = false;
        let mut inner_type: Option<&Bound<'_, PyAny>> = None;
        for a in &args_vec {
            if is_none_type(a) {
                has_none = true;
            } else if inner_type.is_none() {
                inner_type = Some(a);
            }
        }
        if has_none {
            if let Some(inner) = inner_type {
                return infer_json_schema_type(inner);
            }
        }

        if origin_name == "list" {
            return json!({ "type": "array", "items": { "type": "string" } });
        }
        if origin_name == "dict" {
            return json!({ "type": "object" });
        }
    }

    let type_name: String = annotation
        .getattr("__name__")
        .and_then(|n| n.extract())
        .unwrap_or_else(|_| "string".to_string());

    let json_type = match type_name.as_str() {
        "int" => "integer",
        "float" => "number",
        "bool" => "boolean",
        "list" => "array",
        "dict" => "object",
        _ => "string",
    };

    json!({ "type": json_type })
}