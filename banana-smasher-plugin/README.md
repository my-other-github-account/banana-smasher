# banana-smasher-plugin

Native stock-vLLM integration for Banana Smasher model exports.

Install the plugin and its pinned Linux ARM64 serving dependencies from the
Banana Smasher wheelhouse, then use the ordinary vLLM command:

```bash
python -m pip install \
  --extra-index-url https://YOUR-BANANA-WHEELHOUSE/simple \
  banana-smasher-plugin==0.2.0
vllm serve /path/to/exported-model
```

No Banana Smasher repository checkout, serving wrapper, or Banana-specific
vLLM flags are required on the serving host. The
`vllm.general_plugins` entry point recognizes the export's
`quant_method=banana_smasher` identity, applies its versioned runtime profile,
registers the native quantization implementation, and returns control to stock
vLLM.

Explicit vLLM CLI options remain authoritative over plugin defaults.
