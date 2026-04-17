# Experiment Folder Structure Implementation

## Timestamp Format
- Format: `{experiment_type}_{MMDD.HHMM.SS}`
- Example: `single_agent_0417.1430.45` (April 17, 14:30, 45 seconds)

## Folder Structure

### Single-Agent Experiments
```
results/experiments/single_agent_MMDD.HHMM.SS/
├── run1/
│   ├── portfolios/
│   │   ├── buffett.json
│   │   ├── cathie_wood.json
│   │   ├── ray_dalio.json
│   │   ├── ben_graham.json
│   │   └── joel_greenblatt.json
│   ├── experiment_log_buffett.json
│   ├── experiment_log_cathie_wood.json
│   ├── experiment_log_ray_dalio.json
│   ├── experiment_log_ben_graham.json
│   └── experiment_log_joel_greenblatt.json
├── run2/
│   ├── portfolios/
│   │   └── (same portfolio files)
│   └── (same log files)
└── run3/
    └── (same structure)
```

### Multi-Agent Experiments
```
results/experiments/multi_agent_MMDD.HHMM.SS/
├── run1/
│   ├── portfolios/
│   │   ├── majority_vote.json
│   │   ├── average_size.json
│   │   └── llm_manager.json
│   ├── experiment_log_buffett.json          (Pass 1 & 2 logs for buffett)
│   ├── experiment_log_cathie_wood.json      (Pass 1 & 2 logs for cathie_wood)
│   ├── experiment_log_ray_dalio.json
│   ├── experiment_log_ben_graham.json
│   ├── experiment_log_joel_greenblatt.json
│   └── experiment_log_coordinator.json      (decision summary + coordination events)
├── run2/
│   ├── portfolios/
│   │   └── (same coordination mechanism files)
│   └── (same agent + coordinator log files)
└── run3/
    └── (same structure)
```

### Key Features
- Each run is isolated in its own folder with separate logs and portfolios
- Timestamped main experiment folder clearly indicates when experiment was run
- **Per-agent logging**: Each agent gets its own log file (single-agent and multi-agent both use this pattern)
- **Coordinator events**: Decision summary and coordination results logged separately in multi-agent
- Backward compatible - default paths still work if parameters not provided
- Thread-safe for parallel execution with separate log files per agent
- Multi-agent logs isolated from single-agent experiment folder (when both run)
