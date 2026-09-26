//! Opaque conversation capability with an explicit owning device.

use uuid::Uuid;

pub(super) const HEADER: &str = "x-inference-conversation-id";

pub(super) struct ConversationId {
    owner: Uuid,
    value: String,
}

impl ConversationId {
    pub fn new(owner: Uuid) -> Self {
        let value = format!("{owner}.{}", Uuid::new_v4());
        Self { owner, value }
    }

    pub fn parse(value: &str) -> Option<Self> {
        let (owner_text, session_text) = value.split_once('.')?;
        let owner = Uuid::parse_str(owner_text).ok()?;
        let session = Uuid::parse_str(session_text).ok()?;
        if owner.to_string() != owner_text || session.to_string() != session_text {
            return None;
        }
        Some(Self {
            owner,
            value: value.to_owned(),
        })
    }

    pub fn owner(&self) -> Uuid {
        self.owner
    }

    pub fn as_str(&self) -> &str {
        &self.value
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_are_random_and_have_one_canonical_owner() {
        let owner = Uuid::new_v4();
        let first = ConversationId::new(owner);
        let second = ConversationId::new(owner);
        assert_ne!(first.as_str(), second.as_str());
        assert_eq!(
            ConversationId::parse(first.as_str()).unwrap().owner(),
            owner
        );
        assert!(ConversationId::parse("not-a-conversation").is_none());
        assert!(ConversationId::parse(&first.as_str().to_uppercase()).is_none());
    }
}
