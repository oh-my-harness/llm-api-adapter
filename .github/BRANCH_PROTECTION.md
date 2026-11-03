# Main branch protection

## Status

This repository contains the desired state for the existing `main_protect`
ruleset (ID `17346231`). It is not proof that the remote ruleset has already
been changed, and an administrator must apply it manually.

This pipeline did not push, create a pull request, or call any ruleset
`PUT`/`PATCH` endpoint. The remote repository remains unchanged until an
administrator completes the steps below.

## Desired state

[`rulesets/main-protection.json`](rulesets/main-protection.json) is the
complete request body for GitHub's "Update a repository ruleset" endpoint. It:

- makes the ruleset active;
- keeps deletion, non-fast-forward, and required-linear-history protection;
- keeps every existing pull request rule parameter and the rebase-only merge
  method;
- requires the existing `Check`, `wasm32 check`, and `Coverage` checks;
- sets `bypass_actors` to an empty array.

The payload contains request fields only. Response-only values such as `id`,
`source`, `node_id`, timestamps, and `_links` are intentionally omitted.

## Apply manually

Use an account with repository ruleset administration permission. The pipeline
does not provide, save, or use credentials for this operation. From the
repository root, confirm that `gh` is authenticated and inspect the current
ruleset before replacing it:

```sh
gh auth status
gh api --method GET repos/oh-my-harness/llm-api-adapter/rulesets/17346231
```

Apply the desired state with a full ruleset update:

```sh
gh api --method PUT repos/oh-my-harness/llm-api-adapter/rulesets/17346231 --input .github/rulesets/main-protection.json
```

## Verify after applying

Read the same ruleset ID and inspect the enforcement state, bypass actors, and
required check contexts:

```sh
gh api --method GET repos/oh-my-harness/llm-api-adapter/rulesets/17346231 --jq '{enforcement, bypass_actors, pull_request: ([.rules[] | select(.type == "pull_request") | .parameters][0]), required_checks: ([.rules[] | select(.type == "required_status_checks") | .parameters.required_status_checks[].context] | sort)}'
```

The relevant part of the response must show:

```json
{
  "enforcement": "active",
  "bypass_actors": [],
  "pull_request": {
    "allowed_merge_methods": ["rebase"],
    "dismiss_stale_reviews_on_push": false,
    "dismissal_restriction": {
      "allowed_actors": [],
      "enabled": false
    },
    "require_code_owner_review": false,
    "require_extra_approval_for_unattributed_changes": true,
    "require_last_push_approval": false,
    "required_approving_review_count": 0,
    "required_review_thread_resolution": false,
    "required_reviewers": []
  },
  "required_checks": [
    "Check",
    "Coverage",
    "wasm32 check"
  ]
}
```

Also confirm that the returned rules still include deletion,
non-fast-forward, required linear history, and the existing pull request rule
with `allowed_merge_methods` set to `["rebase"]`. Do not describe the protection
as enabled until this GET verification succeeds.
