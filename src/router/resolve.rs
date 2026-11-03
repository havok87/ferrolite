//! Requested model name to model group: exact names, `model_group_alias`,
//! then the longest matching trailing-`*` group.

use std::collections::BTreeMap;

use crate::config::Config;

/// The group a requested model name resolved to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Resolved {
    /// The name the client asked for.
    pub requested: String,
    /// The `model_name` of the group (for wildcards, the pattern itself).
    pub group: String,
    /// Indexes into the router's deployment list.
    pub candidates: Vec<usize>,
    /// For a wildcard group, the text the `*` matched.
    pub capture: Option<String>,
}

pub(crate) struct Groups {
    exact: BTreeMap<String, Vec<usize>>,
    /// `(prefix, group name, deployments)`, longest prefix first.
    wildcards: Vec<(String, String, Vec<usize>)>,
    aliases: BTreeMap<String, String>,
}

impl Groups {
    pub(crate) fn new(config: &Config) -> Self {
        let mut exact: BTreeMap<String, Vec<usize>> = BTreeMap::new();
        let mut wild: BTreeMap<String, Vec<usize>> = BTreeMap::new();
        for (i, d) in config.deployments.iter().enumerate() {
            let target = if d.model_name.ends_with('*') {
                &mut wild
            } else {
                &mut exact
            };
            target.entry(d.model_name.clone()).or_default().push(i);
        }
        let mut wildcards: Vec<_> = wild
            .into_iter()
            .map(|(name, members)| (name.trim_end_matches('*').to_owned(), name, members))
            .collect();
        wildcards.sort_by(|a, b| b.0.len().cmp(&a.0.len()).then_with(|| a.0.cmp(&b.0)));
        let aliases = config
            .router
            .model_group_alias
            .iter()
            .map(|(alias, a)| (alias.clone(), a.target.clone()))
            .collect();
        Self {
            exact,
            wildcards,
            aliases,
        }
    }

    pub(crate) fn resolve(&self, requested: &str) -> Option<Resolved> {
        let target = self
            .aliases
            .get(requested)
            .map_or(requested, String::as_str);
        let mut resolved = self.resolve_group(target)?;
        resolved.requested = requested.to_owned();
        Some(resolved)
    }

    fn resolve_group(&self, name: &str) -> Option<Resolved> {
        if let Some(members) = self.exact.get(name) {
            return Some(Resolved {
                requested: name.to_owned(),
                group: name.to_owned(),
                candidates: members.clone(),
                capture: None,
            });
        }
        self.wildcards.iter().find_map(|(prefix, group, members)| {
            let capture = name.strip_prefix(prefix.as_str())?;
            (!capture.is_empty() && !capture.contains('*')).then(|| Resolved {
                requested: name.to_owned(),
                group: group.clone(),
                candidates: members.clone(),
                capture: Some(capture.to_owned()),
            })
        })
    }
}

/// The model name to send upstream: a trailing `*` in the deployment's
/// model is replaced by what the group's `*` matched.
pub(crate) fn upstream_model(configured: &str, capture: Option<&str>) -> String {
    match (configured.strip_suffix('*'), capture) {
        (Some(prefix), Some(capture)) => format!("{prefix}{capture}"),
        _ => configured.to_owned(),
    }
}
