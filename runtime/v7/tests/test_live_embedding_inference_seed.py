from pathlib import Path


def test_no_grad_fast_forward_reseeds_from_live_model_embedding():
    source = (Path(__file__).parents[1] / "runner/base_binrepair_e2e.py").read_text()
    start = source.index("def fast_forward(")
    end = source.index("\ndef loss_window(", start)
    body = source[start:end]

    embed = body.index("embeds = m.model.embed_tokens(ids)")
    guard = body.index("if not requires_grad:", embed)
    reseed = body.index("hidden = embeds.unsqueeze(2).expand(", guard)
    first_layer = body.index("for Li in range(", reseed)
    assert embed < guard < reseed < first_layer
    assert "config=config, inputs_embeds=embeds" in body
    assert "m.model.embed_tokens(ids[:1])" not in body
