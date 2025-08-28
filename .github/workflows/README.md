flowchart TB
  %% Triggers
  A[Push / Pull Request]:::evt --> B[lint.yml<br/>Lint & Static Analysis]
  A --> C[ci.yml<br/>Unit & Integration Tests]
  A --> D[security.yml<br/>Vuln & Secret Scan]

  %% CI fan-in / status gate
  B --> E{All checks passing?}
  C --> E
  D --> E

  %% Submission
  E -- yes --> F[workflow_dispatch: submission.yml<br/>Predict → Validate → Package<br/>(dry-run by default)]
  E -- no --> X[Fail fast<br/>Block merge]:::fail

  %% Artifacts from CI
  C --> G[[Artifacts:<br/>• test reports<br/>• mini-AOI outputs<br/>• Docker image (optional)]]
  B --> G
  D --> G

  %% Submission branches/guards
  F --> H{Branch = main<br/>+ Secrets present<br/>+ submit=yes?}
  H -- no --> I[Create bundle only:<br/>submission.csv, manifest.json,<br/>candidate dossiers]
  H -- yes --> J[Kaggle Submit<br/>(CLI/API)]:::ship

  %% End products
  I --> K[[Release Artifacts:<br/>• submission.csv<br/>• manifest.json<br/>• /outputs/ dossiers]]
  J --> L[[Kaggle Leaderboard<br/>(public/private)]]

  %% Styling
  classDef evt fill:#eef,stroke:#88f,color:#000,stroke-width:1px;
  classDef fail fill:#fee,stroke:#f55,color:#600,stroke-width:1px;
  classDef ship fill:#efe,stroke:#5a5,color:#060,stroke-width:1px;
