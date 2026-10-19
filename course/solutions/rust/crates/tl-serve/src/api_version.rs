// OpenAI subset version selection for the Rust engine HTTP boundary.
// This module returns response metadata only; the server remains a separate
// process from the Go gateway.
use std::fmt;
#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub enum ApiVersion { V1, V2 }
#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub enum VersionError { Unsupported, Sunset }
impl fmt::Display for VersionError { fn fmt(&self,f:&mut fmt::Formatter<'_>)->fmt::Result { write!(f,"{self:?}") } }
impl std::error::Error for VersionError {}
impl ApiVersion {
    pub fn number(self)->&'static str {
        match self {Self::V1=>"1",Self::V2=>"2"} 
}
}
/// Resolve an explicit request header or the deployment default. Only v1
/// and v2 are legal. A sunset v1 request is rejected before dispatch.
pub fn select(header:Option<&str>,default:ApiVersion,now_unix:i64,sunset_unix:i64)->Result<ApiVersion,VersionError>{
    let v=match header {None=>default,Some("1")=>ApiVersion::V1,Some("2")=>ApiVersion::V2,Some(_)=>return Err(VersionError::Unsupported)};
    if v==ApiVersion::V1 && sunset_unix>0 && now_unix>=sunset_unix {return Err(VersionError::Sunset)} Ok(v)

}
/// Headers shared by unary and streaming responses. Deprecated and Sunset
/// are attached while v1 remains available, then v1 receives HTTP 410.
pub fn response_headers(v:ApiVersion,deprecated_unix:i64,sunset_http_date:&str)->Vec<(&'static str,String)>{
    let mut h=vec![("X-TL-API-Version",v.number().to_string())];
    if v==ApiVersion::V1 {h.push(("Deprecation",format!("@{deprecated_unix}")));h.push(("Sunset",sunset_http_date.to_string()));} h

}
/// v2 uses more specific capacity error names; v1 preserves its original code.
pub fn error_code(v:ApiVersion,code:&str)->&str {
        if v==ApiVersion::V1{return code} match code {"rate_limit_exceeded"=>"requests_limit_exceeded","no_capacity"=>"model_draining",other=>other} 
}
/// v2 requires cached-token usage. Older engines can report zero when the
/// cache metric is unavailable, while the gateway records the version.
pub fn cached_tokens(v:ApiVersion,reported:Option<u32>)->Option<u32> {
        if v==ApiVersion::V2 {Some(reported.unwrap_or(0))} else {None} 
}
