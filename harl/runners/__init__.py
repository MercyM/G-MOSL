"""Runner registry."""
from harl.runners.on_policy_ha_runner import OnPolicyHARunner
from harl.runners.on_policy_ma_runner import OnPolicyMARunner
from harl.runners.off_policy_ha_runner import OffPolicyHARunner
from harl.runners.off_policy_ma_runner import OffPolicyMARunner
from harl.runners.on_policy_ma_graph_runner import OnPolicyMAGraphRunner
from harl.runners.on_policy_ha_graph_runner import OnPolicyHAGraphRunner

RUNNER_REGISTRY = {
    "happo": OnPolicyHARunner,
    "hatrpo": OnPolicyHARunner,
    "haa2c": OnPolicyHARunner,
    "haddpg": OffPolicyHARunner,
    "hatd3": OffPolicyHARunner,
    "hasac": OffPolicyHARunner,
    "had3qn": OffPolicyHARunner,
    "maddpg": OffPolicyMARunner,
    "matd3": OffPolicyMARunner,
    "mappo": OnPolicyMARunner,
    "gmosl": OnPolicyMAGraphRunner,
    "happo_graph": OnPolicyHAGraphRunner,
}
