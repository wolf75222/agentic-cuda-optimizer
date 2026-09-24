import json
from functools import partial

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from openai import OpenAI

from agent_tools import create_tools
from input_generation import prepare_inputs
from nvidia_research import research_nvidia
from nsight_profiler import profile_candidate
from nodes import (
    propose, save_kernel, evaluate_candidate, generate_reference,
    generate_initial_kernel, load_reference_kernel, load_initial_kernel,
    execute_reference, prepare_launches, read_kernel_proposal,
    route_workload, route_reference_result, route_kernel_proposal,
    choose_next, finalize_result, prepare_signature,
)
from optimizer_config import OptimizerConfig
from state import OptimizerState
from use_harness import HarnessRuner


def build_graph(client: OpenAI, runner: HarnessRuner, config: OptimizerConfig):
    builder = StateGraph(OptimizerState)
    tools = create_tools(runner, use_nsight=config.use_nsight)
    model = ChatOpenAI(model=config.model, reasoning_effort=config.reasoning_effort,
                       max_retries=config.max_retries, timeout=120)
    model_with_tools = model.bind_tools(tools)
    tool_node = ToolNode(tools, handle_tool_errors=True)

    def execute_tools(state: OptimizerState) -> dict:
        result = tool_node.invoke(state)
        path = state["session_dir"] / f"tools-{state['kernel_role']}-{state['iteration']}-{state['model_calls']}.json"
        path.write_text(json.dumps([message.model_dump(mode="json") for message in result["messages"]],
                                   indent=2), encoding="utf-8")
        return result

    def call_model(state: OptimizerState) -> dict:
        if state["model_calls"] >= 8 * (config.max_retries + 1):
            return {"error": "Model/tool call budget exhausted for this proposal"}
        response = model_with_tools.invoke(state["messages"])
        call_number = state["model_calls"] + 1
        path = state["session_dir"] / f"model-{state['kernel_role']}-{state['iteration']}-{call_number}.json"
        path.write_text(json.dumps(response.model_dump(mode="json"), indent=2), encoding="utf-8")
        return {"messages": [response], "model_calls": call_number}

    def route_model(state: OptimizerState) -> str:
        if state["error"]:
            return "finish"
        return tools_condition(state)

    builder.add_node("call_model", call_model)
    builder.add_node("tools", execute_tools)
    builder.add_node("prepare_signature", partial(prepare_signature, client=client))
    builder.add_node("research_nvidia", partial(research_nvidia, client=client))
    builder.add_node("prepare_inputs", partial(prepare_inputs, client=client, runner=runner))
    builder.add_node("propose", propose)
    builder.add_node("repair_candidate", partial(propose, repair=True))
    builder.add_node("save_kernel", save_kernel)
    builder.add_node("evaluate_candidate", partial(evaluate_candidate, runner=runner))
    if config.use_nsight:
        builder.add_node("profile_candidate", partial(profile_candidate, runner=runner))
    builder.add_node("generate_reference", partial(generate_reference, client=client))
    builder.add_node("generate_initial_kernel", partial(generate_initial_kernel, client=client))
    builder.add_node("load_reference_kernel", load_reference_kernel)
    builder.add_node("load_initial_kernel", load_initial_kernel)
    builder.add_node("execute_reference", partial(execute_reference, runner=runner))
    builder.add_node("prepare_launches", prepare_launches)
    builder.add_node("read_kernel_proposal", read_kernel_proposal)
    builder.add_node("finalize_result", finalize_result)

    builder.add_edge(START, "prepare_signature")
    builder.add_edge("prepare_signature", "research_nvidia")
    builder.add_edge("research_nvidia", "prepare_inputs")
    routes = {
        "generate_reference": "generate_reference",
        "generate_initial_kernel": "generate_initial_kernel",
        "ready": "load_reference_kernel",
    }
    for source in ("prepare_inputs", "generate_reference", "generate_initial_kernel"):
        builder.add_conditional_edges(source, route_workload, routes)

    builder.add_edge("load_reference_kernel", "prepare_launches")
    builder.add_edge("load_initial_kernel", "prepare_launches")
    builder.add_conditional_edges("execute_reference", route_reference_result, {
        "continue": "load_initial_kernel", "finish": "finalize_result",
    })
    builder.add_edge("prepare_launches", "call_model")
    builder.add_edge("propose", "call_model")
    builder.add_edge("repair_candidate", "call_model")
    builder.add_conditional_edges("call_model", route_model, {
        "tools": "tools", "__end__": "read_kernel_proposal", "finish": "finalize_result",
    })
    builder.add_edge("tools", "call_model")
    builder.add_conditional_edges("read_kernel_proposal", route_kernel_proposal, {
        "reference": "execute_reference",
        "candidate": "save_kernel",
        "retry": "call_model",
        "finish": "finalize_result",
    })
    builder.add_edge("save_kernel", "evaluate_candidate")
    next_step = "evaluate_candidate"
    if config.use_nsight:
        builder.add_edge("evaluate_candidate", "profile_candidate")
        next_step = "profile_candidate"
    builder.add_conditional_edges(next_step, choose_next, {
        "repeat": "propose", "repair": "repair_candidate", "finish": "finalize_result",
    })
    builder.add_edge("finalize_result", END)
    return builder.compile()
