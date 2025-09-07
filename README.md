crewai-data-discovery/
│
├── README.md
├── requirements.txt
├── pyproject.toml
├── setup.py
├── .gitignore
├── .env.example
├── .pre-commit-config.yaml
├── docker-compose.yml
├── Dockerfile
│
├── src/
│   └── data_discovery/
│       ├── __init__.py
│       ├── main.py
│       ├── orchestrator.py
│       │
│       ├── agents/
│       │   ├── __init__.py
│       │   ├── base_agent.py
│       │   ├── orchestrator_agent.py
│       │   ├── technical_analyst_agent.py
│       │   ├── business_analyst_agent.py
│       │   ├── data_scientist_agent.py
│       │   ├── query_specialist_agent.py
│       │   ├── qa_agent.py
│       │   ├── governance_agent.py
│       │   └── ux_agent.py
│       │
│       ├── tools/
│       │   ├── __init__.py
│       │   ├── core/
│       │   │   ├── __init__.py
│       │   │   ├── state_manager.py
│       │   │   ├── error_handler.py
│       │   │   ├── progress_tracker.py
│       │   │   └── connection_validator.py
│       │   │
│       │   ├── analysis/
│       │   │   ├── __init__.py
│       │   │   ├── schema_inspector.py
│       │   │   ├── metadata_extractor.py
│       │   │   ├── data_profiler.py
│       │   │   ├── quality_scorer.py
│       │   │   ├── distribution_analyzer.py
│       │   │   ├── domain_classifier.py
│       │   │   ├── business_glossary.py
│       │   │   └── concept_mapper.py
│       │   │
│       │   ├── query/
│       │   │   ├── __init__.py
│       │   │   ├── sql_generator.py
│       │   │   ├── query_optimizer.py
│       │   │   ├── business_question_bank.py
│       │   │   ├── validation_suite.py
│       │   │   ├── syntax_checker.py
│       │   │   ├── logic_validator.py
│       │   │   └── performance_estimator.py
│       │   │
│       │   ├── governance/
│       │   │   ├── __init__.py
│       │   │   ├── sensitive_data_detector.py
│       │   │   ├── compliance_scanner.py
│       │   │   ├── masking_recommender.py
│       │   │   └── pii_patterns.py
│       │   │
│       │   └── ui/
│       │       ├── __init__.py
│       │       ├── gradio_catalog.py
│       │       ├── ui_generator.py
│       │       ├── search_engine.py
│       │       └── filter_manager.py
│       │
│       ├── models/
│       │   ├── __init__.py
│       │   ├── catalog.py
│       │   ├── schema_metadata.py
│       │   ├── query_result.py
│       │   ├── governance_report.py
│       │   └── business_domain.py
│       │
│       ├── core/
│       │   ├── __init__.py
│       │   ├── config.py
│       │   ├── exceptions.py
│       │   ├── logging_config.py
│       │   ├── state_manager.py
│       │   └── database_connector.py
│       │
│       ├── security/
│       │   ├── __init__.py
│       │   ├── credential_manager.py
│       │   ├── access_control.py
│       │   ├── encryption_utils.py
│       │   └── audit_logger.py
│       │
│       ├── monitoring/
│       │   ├── __init__.py
│       │   ├── metrics_collector.py
│       │   ├── health_checker.py
│       │   ├── performance_monitor.py
│       │   └── alerting.py
│       │
│       └── utils/
│           ├── __init__.py
│           ├── data_utils.py
│           ├── string_utils.py
│           ├── date_utils.py
│           └── validation_utils.py
│
├── config/
│   ├── __init__.py
│   ├── base_config.py
│   ├── development.py
│   ├── production.py
│   ├── testing.py
│   ├── snowflake_config.py
│   └── gradio_config.py
│
├── data/
│   ├── schemas/
│   │   ├── catalog_schema.json
│   │   ├── metadata_schema.json
│   │   └── governance_schema.json
│   │
│   ├── templates/
│   │   ├── business_questions/
│   │   │   ├── trend_analysis.json
│   │   │   ├── top_n_queries.json
│   │   │   ├── comparison_queries.json
│   │   │   ├── distribution_queries.json
│   │   │   └── anomaly_detection.json
│   │   │
│   │   └── sql_templates/
│   │       ├── basic_aggregations.sql
│   │       ├── time_series.sql
│   │       ├── ranking_queries.sql
│   │       └── analytical_functions.sql
│   │
│   ├── samples/
│   │   ├── sample_schema.json
│   │   ├── sample_catalog.json
│   │   └── sample_governance_report.json
│   │
│   └── reference/
│       ├── business_glossary.json
│       ├── domain_mappings.json
│       └── pii_patterns.json
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── fixtures/
│   │   ├── __init__.py
│   │   ├── sample_data.py
│   │   ├── mock_snowflake.py
│   │   └── test_schemas.py
│   │
│   ├── unit/
│   │   ├── __init__.py
│   │   ├── agents/
│   │   │   ├── test_orchestrator_agent.py
│   │   │   ├── test_technical_analyst_agent.py
│   │   │   ├── test_business_analyst_agent.py
│   │   │   ├── test_data_scientist_agent.py
│   │   │   ├── test_query_specialist_agent.py
│   │   │   ├── test_qa_agent.py
│   │   │   ├── test_governance_agent.py
│   │   │   └── test_ux_agent.py
│   │   │
│   │   ├── tools/
│   │   │   ├── test_schema_inspector.py
│   │   │   ├── test_data_profiler.py
│   │   │   ├── test_sql_generator.py
│   │   │   ├── test_validation_suite.py
│   │   │   ├── test_sensitive_data_detector.py
│   │   │   └── test_gradio_catalog.py
│   │   │
│   │   ├── core/
│   │   │   ├── test_state_manager.py
│   │   │   ├── test_error_handler.py
│   │   │   ├── test_config.py
│   │   │   └── test_database_connector.py
│   │   │
│   │   └── security/
│   │       ├── test_credential_manager.py
│   │       ├── test_access_control.py
│   │       └── test_audit_logger.py
│   │
│   ├── integration/
│   │   ├── __init__.py
│   │   ├── test_snowflake_integration.py
│   │   ├── test_agent_collaboration.py
│   │   ├── test_pipeline_execution.py
│   │   └── test_gradio_interface.py
│   │
│   ├── e2e/
│   │   ├── __init__.py
│   │   ├── test_full_discovery_pipeline.py
│   │   ├── test_catalog_generation.py
│   │   └── test_ui_workflows.py
│   │
│   └── performance/
│       ├── __init__.py
│       ├── test_scalability.py
│       ├── test_query_performance.py
│       └── benchmark_agents.py
│
├── docs/
│   ├── README.md
│   ├── architecture.md
│   ├── deployment.md
│   ├── security.md
│   ├── monitoring.md
│   ├── troubleshooting.md
│   │
│   ├── api/
│   │   ├── agents.md
│   │   ├── tools.md
│   │   ├── models.md
│   │   └── configuration.md
│   │
│   ├── user_guide/
│   │   ├── getting_started.md
│   │   ├── configuration.md
│   │   ├── using_catalog.md
│   │   └── best_practices.md
│   │
│   └── examples/
│       ├── basic_usage.py
│       ├── custom_agents.py
│       ├── advanced_configuration.py
│       └── integration_examples.py
│
├── scripts/
│   ├── __init__.py
│   ├── setup_database.py
│   ├── generate_sample_data.py
│   ├── run_discovery.py
│   ├── export_catalog.py
│   ├── health_check.py
│   ├── backup_state.py
│   └── migrate_schema.py
│
├── deployment/
│   ├── docker/
│   │   ├── Dockerfile.prod
│   │   ├── Dockerfile.dev
│   │   ├── docker-compose.prod.yml
│   │   └── docker-compose.dev.yml
│   │
│   ├── kubernetes/
│   │   ├── namespace.yaml
│   │   ├── deployment.yaml
│   │   ├── service.yaml
│   │   ├── configmap.yaml
│   │   ├── secret.yaml
│   │   └── ingress.yaml
│   │
│   ├── terraform/
│   │   ├── main.tf
│   │   ├── variables.tf
│   │   ├── outputs.tf
│   │   ├── provider.tf
│   │   └── modules/
│   │       ├── snowflake/
│   │       ├── monitoring/
│   │       └── security/
│   │
│   └── helm/
│       ├── Chart.yaml
│       ├── values.yaml
│       ├── values-prod.yaml
│       ├── values-dev.yaml
│       └── templates/
│           ├── deployment.yaml
│           ├── service.yaml
│           ├── configmap.yaml
│           └── ingress.yaml
│
├── monitoring/
│   ├── prometheus/
│   │   ├── prometheus.yml
│   │   └── rules/
│   │       ├── agents.yml
│   │       ├── performance.yml
│   │       └── errors.yml
│   │
│   ├── grafana/
│   │   ├── dashboards/
│   │   │   ├── agent_performance.json
│   │   │   ├── catalog_metrics.json
│   │   │   ├── snowflake_connections.json
│   │   │   └── user_activity.json
│   │   │
│   │   └── provisioning/
│   │       ├── datasources/
│   │       └── dashboards/
│   │
│   └── alerts/
│       ├── slack_webhook.py
│       ├── email_notifications.py
│       └── pagerduty_integration.py
│
├── logs/
│   ├── .gitkeep
│   ├── agents/
│   ├── errors/
│   ├── performance/
│   └── audit/
│
├── cache/
│   ├── .gitkeep
│   ├── schemas/
│   ├── queries/
│   ├── profiles/
│   └── governance/
│
└── .github/
    ├── workflows/
    │   ├── ci.yml
    │   ├── cd.yml
    │   ├── security-scan.yml
    │   ├── performance-test.yml
    │   └── docs-deploy.yml
    │
    ├── ISSUE_TEMPLATE/
    │   ├── bug_report.md
    │   ├── feature_request.md
    │   └── performance_issue.md
    │
    ├── PULL_REQUEST_TEMPLATE.md
    ├── CONTRIBUTING.md
    ├── CODE_OF_CONDUCT.md
    └── SECURITY.md