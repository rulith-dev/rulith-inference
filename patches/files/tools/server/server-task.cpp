#include "server-task.h"

#include "build-info.h"
#include "server-chat.h"
#include "chat.h"
#include "common.h"
#include "json-schema-to-grammar.h"
#include "llama.h"
#include "sampling.h"
#include "speculative.h"
#include "server-common.h"

#include <sstream>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <future>
#include <set>
#include <system_error>

// XXH3 names the disk tier's chunks and checkpoints; inlined here, so nothing new to link
#define XXH_INLINE_ALL
#include "vendor/hash/xxhash/xxhash.h"
#ifdef _WIN32
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <windows.h>
#endif

//
// task_params
//

json task_params::format_logit_bias(const std::vector<llama_logit_bias> & logit_bias) const {
    json data = json::array();
    for (const auto & lb : logit_bias) {
        data.push_back(json{
            {"bias", lb.bias},
            {"token", lb.token},
        });
    }
    return data;
}

json task_params::to_json(bool only_metrics) const {
    std::vector<std::string> samplers;
    samplers.reserve(sampling.samplers.size());
    for (const auto & sampler : sampling.samplers) {
        samplers.emplace_back(common_sampler_type_to_str(sampler));
    }

    json lora = json::array();
    for (auto & it : this->lora) {
        lora.push_back({{"id", it.first}, {"scale", it.second}});
    }

    if (only_metrics) {
        return json {
            {"seed",                      sampling.seed},
            {"temperature",               sampling.temp},
            {"dynatemp_range",            sampling.dynatemp_range},
            {"dynatemp_exponent",         sampling.dynatemp_exponent},
            {"top_k",                     sampling.top_k},
            {"top_p",                     sampling.top_p},
            {"min_p",                     sampling.min_p},
            {"top_n_sigma",               sampling.top_n_sigma},
            {"xtc_probability",           sampling.xtc_probability},
            {"xtc_threshold",             sampling.xtc_threshold},
            {"typical_p",                 sampling.typ_p},
            {"repeat_last_n",             sampling.penalty_last_n},
            {"repeat_penalty",            sampling.penalty_repeat},
            {"presence_penalty",          sampling.penalty_present},
            {"frequency_penalty",         sampling.penalty_freq},
            {"dry_multiplier",            sampling.dry_multiplier},
            {"dry_base",                  sampling.dry_base},
            {"dry_allowed_length",        sampling.dry_allowed_length},
            {"dry_penalty_last_n",        sampling.dry_penalty_last_n},
            {"mirostat",                  sampling.mirostat},
            {"mirostat_tau",              sampling.mirostat_tau},
            {"mirostat_eta",              sampling.mirostat_eta},
            {"adaptive_target",           sampling.adaptive_target},
            {"adaptive_decay",            sampling.adaptive_decay},
            {"max_tokens",                n_predict},
            {"n_predict",                 n_predict}, // TODO: deduplicate?
            {"n_keep",                    n_keep},
            {"n_discard",                 n_discard},
            {"ignore_eos",                sampling.ignore_eos},
            {"stream",                    stream},
            {"n_probs",                   sampling.n_probs},
            {"min_keep",                  sampling.min_keep},
            {"chat_format",               common_chat_format_name(chat_format)},
            {"reasoning_format",          common_reasoning_format_name(reasoning_format)},
            {"reasoning_in_content",      stream && reasoning_format == COMMON_REASONING_FORMAT_DEEPSEEK_LEGACY},
            {"generation_prompt",         sampling.generation_prompt},
            {"samplers",                  samplers},
            {"speculative.types",         common_speculative_type_name_str(speculative.types)},
            {"timings_per_token",         timings_per_token},
            {"post_sampling_probs",       post_sampling_probs},
            {"backend_sampling",          sampling.backend_sampling},
            {"lora",                      lora},
        };
    }

    auto grammar_triggers = json::array();
    for (const auto & trigger : sampling.grammar_triggers) {
        server_grammar_trigger ct(trigger);
        grammar_triggers.push_back(ct.to_json());
    }

    return json {
        {"seed",                      sampling.seed},
        {"temperature",               sampling.temp},
        {"dynatemp_range",            sampling.dynatemp_range},
        {"dynatemp_exponent",         sampling.dynatemp_exponent},
        {"top_k",                     sampling.top_k},
        {"top_p",                     sampling.top_p},
        {"min_p",                     sampling.min_p},
        {"top_n_sigma",               sampling.top_n_sigma},
        {"xtc_probability",           sampling.xtc_probability},
        {"xtc_threshold",             sampling.xtc_threshold},
        {"typical_p",                 sampling.typ_p},
        {"repeat_last_n",             sampling.penalty_last_n},
        {"repeat_penalty",            sampling.penalty_repeat},
        {"presence_penalty",          sampling.penalty_present},
        {"frequency_penalty",         sampling.penalty_freq},
        {"dry_multiplier",            sampling.dry_multiplier},
        {"dry_base",                  sampling.dry_base},
        {"dry_allowed_length",        sampling.dry_allowed_length},
        {"dry_penalty_last_n",        sampling.dry_penalty_last_n},
        {"dry_sequence_breakers",     sampling.dry_sequence_breakers},
        {"mirostat",                  sampling.mirostat},
        {"mirostat_tau",              sampling.mirostat_tau},
        {"mirostat_eta",              sampling.mirostat_eta},
        {"adaptive_target",           sampling.adaptive_target},
        {"adaptive_decay",            sampling.adaptive_decay},
        {"stop",                      antiprompt},
        {"max_tokens",                n_predict},
        {"n_predict",                 n_predict}, // TODO: deduplicate?
        {"n_keep",                    n_keep},
        {"n_discard",                 n_discard},
        {"ignore_eos",                sampling.ignore_eos},
        {"stream",                    stream},
        {"logit_bias",                format_logit_bias(sampling.logit_bias)},
        {"n_probs",                   sampling.n_probs},
        {"min_keep",                  sampling.min_keep},
        {"grammar",                   common_grammar_value(sampling.grammar)},
        {"grammar_lazy",              sampling.grammar_lazy},
        {"grammar_triggers",          grammar_triggers},
        {"preserved_tokens",          sampling.preserved_tokens},
        {"chat_format",               common_chat_format_name(chat_format)},
        {"reasoning_format",          common_reasoning_format_name(reasoning_format)},
        {"reasoning_in_content",      stream && reasoning_format == COMMON_REASONING_FORMAT_DEEPSEEK_LEGACY},
        {"generation_prompt",         sampling.generation_prompt},
        {"samplers",                  samplers},
        {"speculative.types",         common_speculative_type_name_str(speculative.types)},
        {"timings_per_token",         timings_per_token},
        {"post_sampling_probs",       post_sampling_probs},
        {"backend_sampling",          sampling.backend_sampling},
        {"lora",                      lora},
    };
}

//
// task_result_state
//
task_result_state::task_result_state(common_chat_session session)
    : chat_session(std::move(session))
    , chat_msg(chat_session.msg())
    , oai_resp_id("resp_" + random_string())
    , oai_resp_reasoning_id("rs_" + random_string())
    , oai_resp_message_id("msg_" + random_string()) {
}

common_chat_msg task_result_state::update_chat_msg(
        const common_chat_input & added,
        bool is_partial,
        std::vector<common_chat_msg_diff> & diffs,
        bool filter_tool_calls) {
    auto msg_prv_copy = chat_msg;
    generated_text += added.text;
    auto new_msg = is_partial ? chat_session.feed(added) : chat_session.finish(added);
    if (!new_msg.empty()) {
        new_msg.set_tool_call_ids(generated_tool_call_ids, gen_tool_call_id);
        chat_msg = new_msg;
        auto all_diffs = common_chat_msg_diff::compute_diffs(msg_prv_copy, chat_msg);

        if (!filter_tool_calls) {
            diffs = std::move(all_diffs);
        } else {
            for (auto & d : all_diffs) {
                // If this is a new type of delta, flush all currently pending tool call names
                for (size_t i = 0; i < chat_msg.tool_calls.size(); ++i) {
                    if (sent_tool_call_names.count(i) || chat_msg.tool_calls[i].name.empty()) {
                        continue;
                    }
                    if (d.tool_call_index != i || !d.tool_call_delta.arguments.empty()) {
                        common_chat_msg_diff header;
                        header.tool_call_index      = i;
                        header.tool_call_delta.id   = chat_msg.tool_calls[i].id;
                        header.tool_call_delta.name = chat_msg.tool_calls[i].name;
                        diffs.push_back(std::move(header));
                        sent_tool_call_names.insert(i);
                    }
                }

                if (d.tool_call_index == std::string::npos) {
                    diffs.push_back(std::move(d));
                } else {
                    size_t i = d.tool_call_index;
                    if (sent_tool_call_names.count(i)) {
                        if (!d.tool_call_delta.arguments.empty()) {
                            d.tool_call_delta.name = "";
                            d.tool_call_delta.id   = "";
                            diffs.push_back(std::move(d));
                        }
                    } else {
                        // Not sent yet.
                        if (!d.tool_call_delta.arguments.empty() || !is_partial) {
                            d.tool_call_delta.name = chat_msg.tool_calls[i].name;
                            d.tool_call_delta.id   = chat_msg.tool_calls[i].id;
                            diffs.push_back(std::move(d));
                            sent_tool_call_names.insert(i);
                        } else {
                            // Suppress
                        }
                    }
                }
            }
            // Final check at EOF
            if (!is_partial) {
                for (size_t i = 0; i < chat_msg.tool_calls.size(); ++i) {
                    if (!sent_tool_call_names.count(i) && !chat_msg.tool_calls[i].name.empty()) {
                        common_chat_msg_diff header;
                        header.tool_call_index      = i;
                        header.tool_call_delta.id   = chat_msg.tool_calls[i].id;
                        header.tool_call_delta.name = chat_msg.tool_calls[i].name;
                        diffs.push_back(std::move(header));
                        sent_tool_call_names.insert(i);
                    }
                }
            }
        }
    }
    return chat_msg;
}

//
// result_prompt_progress
//
json result_prompt_progress::to_json() const {
    return json {
        {"total",     total},
        {"cache",     cache},
        {"processed", processed},
        {"time_ms",   time_ms},
    };
}

static inline std::string stop_type_to_str(stop_type type) {
    switch (type) {
        case STOP_TYPE_EOS:   return "eos";
        case STOP_TYPE_WORD:  return "word";
        case STOP_TYPE_LIMIT: return "limit";
        default:              return "none";
    }
}

//
// completion_token_output
//

json completion_token_output::to_json(bool post_sampling_probs) const {
    json probs_for_token = json::array();
    for (const auto & p : probs) {
        std::string txt(p.txt);
        txt.resize(validate_utf8(txt));
        probs_for_token.push_back(json {
            {"id",      p.tok},
            {"token",   txt},
            {"bytes",   str_to_bytes(p.txt)},
            {
                post_sampling_probs ? "prob" : "logprob",
                post_sampling_probs ? p.prob : logarithm(p.prob)
            },
        });
    }
    return probs_for_token;
}

json completion_token_output::probs_vector_to_json(const std::vector<completion_token_output> & probs, bool post_sampling_probs) {
    json out = json::array();
    for (const auto & p : probs) {
        std::string txt(p.text_to_send);
        txt.resize(validate_utf8(txt));
        out.push_back(json {
            {"id",           p.tok},
            {"token",        txt},
            {"bytes",        str_to_bytes(p.text_to_send)},
            {
                post_sampling_probs ? "prob" : "logprob",
                post_sampling_probs ? p.prob : logarithm(p.prob)
            },
            {
                post_sampling_probs ? "top_probs" : "top_logprobs",
                p.to_json(post_sampling_probs)
            },
        });
    }
    return out;
}

float completion_token_output::logarithm(float x) {
    // the JSON library converts -inf to null, so we need to prevent that
    return x == 0.0f ? std::numeric_limits<float>::lowest() : std::log(x);
}

std::vector<unsigned char> completion_token_output::str_to_bytes(const std::string & str) {
    std::vector<unsigned char> bytes;
    for (unsigned char c : str) {
        bytes.push_back(c);
    }
    return bytes;
}

//
// server_task_result_cmpl_final
//
json server_task_result_cmpl_final::to_json() {
    GGML_ASSERT(is_updated && "update() must be called before to_json()");
    switch (res_type) {
        case TASK_RESPONSE_TYPE_NONE:
            return to_json_non_oaicompat();
        case TASK_RESPONSE_TYPE_OAI_CMPL:
            return to_json_oaicompat();
        case TASK_RESPONSE_TYPE_OAI_CHAT:
            return stream ? to_json_oaicompat_chat_stream() : to_json_oaicompat_chat();
        case TASK_RESPONSE_TYPE_OAI_RESP:
            return stream ? to_json_oaicompat_resp_stream() : to_json_oaicompat_resp();
        case TASK_RESPONSE_TYPE_OAI_ASR:
            return to_json_oaicompat_asr();
        case TASK_RESPONSE_TYPE_ANTHROPIC:
            return stream ? to_json_anthropic_stream() : to_json_anthropic();
        default:
            GGML_ASSERT(false && "Invalid task_response_type");
    }
}

json server_task_result_cmpl_final::to_json_non_oaicompat() {
    json res = json {
        {"index",               index},
        {"content",             content.text},
        {"tokens",              tokens},
        {"id_slot",             id_slot},
        {"stop",                true},
        {"model",               oaicompat_model},
        {"tokens_predicted",    n_decoded},
        {"tokens_evaluated",    n_prompt_tokens},
        {"generation_settings", generation_params.to_json()},
        {"prompt",              prompt},
        {"has_new_line",        has_new_line},
        {"truncated",           truncated},
        {"stop_type",           stop_type_to_str(stop)},
        {"stopping_word",       stopping_word},
        {"tokens_cached",       n_tokens_cached},
        {"timings",             stats.to_json()},
    };
    if (!stream && !probs_output.empty()) {
        res["completion_probabilities"] = completion_token_output::probs_vector_to_json(probs_output, post_sampling_probs);
    }
    return response_fields.empty() ? res : json_get_nested_values(response_fields, res);
}

json server_task_result_cmpl_final::usage_json_oaicompat() {
    return json {
        {"completion_tokens", n_decoded},
        {"prompt_tokens",     n_prompt_tokens},
        {"total_tokens",      n_decoded + n_prompt_tokens},
        {"prompt_tokens_details", json { {"cached_tokens", n_prompt_tokens_cache} }},
    };
}

json server_task_result_cmpl_final::to_json_oaicompat() {
    std::time_t t = std::time(0);
    json logprobs = json(nullptr); // OAI default to null
    if (!stream && probs_output.size() > 0) {
        logprobs = json{
            {"content", completion_token_output::probs_vector_to_json(probs_output, post_sampling_probs)},
        };
    }
    json finish_reason = "length";
    if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
        finish_reason = "stop";
    }
    json res = json {
        {"choices",            json::array({
            json{
                {"text",          content.text},
                {"index",         index},
                {"logprobs",      logprobs},
                {"finish_reason", finish_reason},
            }
        })},
        {"created",            t},
        {"model",              oaicompat_model},
        {"system_fingerprint", std::string(llama_build_info())},
        {"object",             "text_completion"},
        {"usage",              usage_json_oaicompat()},
        {"id", oaicompat_cmpl_id}
    };

    // extra fields for debugging purposes
    if (verbose) {
        res["__verbose"] = to_json_non_oaicompat();
    }
    if (stats.is_set()) {
        res["timings"] = stats.to_json();
    }

    return res;
}

json server_task_result_cmpl_final::to_json_oaicompat_chat() {
    std::string finish_reason = "length";
    common_chat_msg msg;
    if (!oaicompat_msg.empty()) {
        msg = oaicompat_msg;
    } else {
        msg.role = "assistant";
        msg.content = content.text;
    }
    if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
        finish_reason = msg.tool_calls.empty() ? "stop" : "tool_calls";
    }

    json choice {
        {"finish_reason", finish_reason},
        {"index", index},
        {"message", msg.to_json_oaicompat()},
    };

    if (!stream && probs_output.size() > 0) {
        choice["logprobs"] = json{
            {"content", completion_token_output::probs_vector_to_json(probs_output, post_sampling_probs)},
        };
    }

    std::time_t t = std::time(0);

    json res = json {
        {"choices",            json::array({choice})},
        {"created",            t},
        {"model",              oaicompat_model},
        {"system_fingerprint", std::string(llama_build_info())},
        {"object",             "chat.completion"},
        {"usage",              usage_json_oaicompat()},
        {"id", oaicompat_cmpl_id}
    };

    // extra fields for debugging purposes
    if (verbose) {
        res["__verbose"] = to_json_non_oaicompat();
    }
    if (stats.is_set()) {
        res["timings"] = stats.to_json();
    }

    return res;
}

json server_task_result_cmpl_final::to_json_oaicompat_chat_stream() {
    std::time_t t = std::time(0);
    std::string finish_reason = "length";
    if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
        finish_reason = oaicompat_msg.tool_calls.empty() ? "stop" : "tool_calls";
    }

    json deltas = json::array();
    for (const auto & diff : oaicompat_msg_diffs) {
        deltas.push_back({
            {"choices", json::array({
                json {
                    {"finish_reason", nullptr},
                    {"index", index},
                    {"delta", server_chat_msg_diff_to_json_oaicompat(diff)},
                },
            })},
            {"created", t},
            {"id", oaicompat_cmpl_id},
            {"model", oaicompat_model},
            {"system_fingerprint", std::string(llama_build_info())},
            {"object", "chat.completion.chunk"},
        });
    }

    deltas.push_back({
        {"choices", json::array({
            json {
                {"finish_reason", finish_reason},
                {"index", index},
                {"delta", json::object()},
            },
        })},
        {"created",            t},
        {"id",                 oaicompat_cmpl_id},
        {"model",              oaicompat_model},
        {"system_fingerprint", std::string(llama_build_info())},
        {"object",             "chat.completion.chunk"},
    });

    if (include_usage) {
        // OpenAI API spec for chat.completion.chunks specifies an empty `choices` array for the last chunk when including usage
        // https://platform.openai.com/docs/api-reference/chat_streaming/streaming#chat_streaming/streaming-choices
        deltas.push_back({
            {"choices", json::array()},
            {"created",            t},
            {"id",                 oaicompat_cmpl_id},
            {"model",              oaicompat_model},
            {"system_fingerprint", std::string(llama_build_info())},
            {"object",             "chat.completion.chunk"},
            {"usage",              usage_json_oaicompat()},
        });
    }

    if (stats.is_set()) {
        deltas.back()["timings"] = stats.to_json();
    }

    // extra fields for debugging purposes
    if (verbose && !deltas.empty()) {
        deltas.front()["__verbose"] = to_json_non_oaicompat();
    }

    return deltas;
}

// strixllama: the client's view of a tool call for the OpenAI Responses API (task_params::resp_tools). A call of a
// custom tool, offered to the model as a function taking one string "input", is a custom_tool_call carrying that input;
// a call of a tool from a namespace names the tool and its namespace; other calls are left as they are.
static const json * resp_tool_entry(const std::shared_ptr<const json> & tools, const std::string & name) {
    if (!tools || !tools->is_object() || !tools->contains(name)) {
        return nullptr;
    }
    return &tools->at(name);
}

static bool resp_tool_is_custom(const std::shared_ptr<const json> & tools, const std::string & name) {
    const json * e = resp_tool_entry(tools, name);
    return e && json_value(*e, "type", std::string()) == "custom";
}

static std::string resp_custom_input(const std::string & arguments) {
    const json args = json::parse_no_throw(arguments);
    if (!args.is_discarded() && args.is_object() && args.contains("input")) {
        return args.at("input").is_string() ? args.at("input").get<std::string>() : args.at("input").dump_safe();
    }
    return arguments;
}

static json resp_tool_call_item(const std::shared_ptr<const json> & tools, const common_chat_tool_call & tool_call) {
    const json * e = resp_tool_entry(tools, tool_call.name);
    json item;
    if (e && json_value(*e, "type", std::string()) == "custom") {
        item = json {
            {"id",      "ctc_" + tool_call.id},
            {"type",    "custom_tool_call"},
            {"status",  "completed"},
            {"call_id", "call_" + tool_call.id},
            {"name",    json_value(*e, "name", tool_call.name)},
            {"input",   resp_custom_input(tool_call.arguments)},
        };
    } else {
        item = json {
            {"id",        "fc_" + tool_call.id},
            {"type",      "function_call"},
            {"status",    "completed"},
            {"arguments", tool_call.arguments},
            {"call_id",   "call_" + tool_call.id},
            {"name",      e ? json_value(*e, "name", tool_call.name) : tool_call.name},
        };
    }
    if (e && e->contains("namespace")) {
        item["namespace"] = e->at("namespace");
    }
    return item;
}

json server_task_result_cmpl_final::to_json_oaicompat_resp() {
    common_chat_msg msg;
    if (!oaicompat_msg.empty()) {
        msg = oaicompat_msg;
    } else {
        msg.role = "assistant";
        msg.content = content.text;
    }

    std::vector<json> output;

    if (msg.reasoning_content != "") {
        output.push_back(json {
            {"id",      "rs_" + random_string()},
            {"summary", json::array()},
            {"type",    "reasoning"},
            {"content", json::array({ json {
                {"text", msg.reasoning_content},
                {"type", "reasoning_text"},
            }})},
            {"encrypted_content", ""},
            {"status",            "completed"},
        });
    }

    if (msg.content != "") {
        output.push_back(json {
            {"content", json::array({ json {
                {"type",        "output_text"},
                {"annotations", json::array()},
                {"logprobs",    json::array()},
                {"text",        msg.content},
            }})},
            {"id",     "msg_" + random_string()},
            {"role",   msg.role},
            {"status", "completed"},
            {"type",   "message"},
        });
    }

    for (const common_chat_tool_call & tool_call : oaicompat_msg.tool_calls) {
        output.push_back(resp_tool_call_item(resp_tools, tool_call));
    }

    std::time_t t = std::time(0);
    json res = {
        {"completed_at", t},
        {"created_at",   t},
        {"id",           oai_resp_id},
        {"model",        oaicompat_model},
        {"object",       "response"},
        {"output",       output},
        {"status",       "completed"},
        {"usage",        json {
            {"input_tokens",  n_prompt_tokens},
            {"output_tokens", n_decoded},
            {"total_tokens",  n_decoded + n_prompt_tokens},
            {"input_tokens_details", json { {"cached_tokens", n_prompt_tokens_cache} }},
        }},
    };

    return res;
}

json server_task_result_cmpl_final::to_json_oaicompat_resp_stream() {
    std::vector<json> server_sent_events;
    std::vector<json> output;
    size_t output_index = 0;

    if (oaicompat_msg.reasoning_content != "") {
        const size_t item_output_index = output_index++;
        const json output_item = json {
            {"id",      oai_resp_reasoning_id},
            {"summary", json::array()},
            {"type",    "reasoning"},
            {"content", json::array({ json {
                {"text", oaicompat_msg.reasoning_content},
                {"type", "reasoning_text"},
            }})},
            {"encrypted_content", ""},
        };

        server_sent_events.push_back(json {
            {"event", "response.output_item.done"},
            {"data", json {
                {"type", "response.output_item.done"},
                {"output_index", item_output_index},
                {"item", output_item}
            }}
        });
        output.push_back(output_item);
    }

    if (oaicompat_msg.content != "") {
        const size_t item_output_index = output_index++;
        server_sent_events.push_back(json {
            {"event", "response.output_text.done"},
            {"data", json {
                {"type",    "response.output_text.done"},
                {"item_id", oai_resp_message_id},
                {"output_index", item_output_index},
                {"content_index", 0},
                {"text",    oaicompat_msg.content}
            }}
        });

        const json content_part = {
            {"type",        "output_text"},
            {"annotations", json::array()},
            {"logprobs",    json::array()},
            {"text",        oaicompat_msg.content}
        };

        server_sent_events.push_back(json {
            {"event", "response.content_part.done"},
            {"data", json {
                {"type",    "response.content_part.done"},
                {"item_id", oai_resp_message_id},
                {"output_index", item_output_index},
                {"content_index", 0},
                {"part",    content_part}
            }}
        });
        const json output_item = {
            {"type",    "message"},
            {"status",  "completed"},
            {"id",      oai_resp_message_id},
            {"content", json::array({content_part})},
            {"role",    "assistant"}
        };

        server_sent_events.push_back(json {
            {"event", "response.output_item.done"},
            {"data", json {
                {"type", "response.output_item.done"},
                {"output_index", item_output_index},
                {"item", output_item}
            }}
        });
        output.push_back(output_item);
    }

    for (const common_chat_tool_call & tool_call : oaicompat_msg.tool_calls) {
        const size_t item_output_index = output_index++;
        const json output_item = resp_tool_call_item(resp_tools, tool_call);
        if (output_item.at("type") == "custom_tool_call") {
            // strixllama: the partial results held a custom call's events back (its arguments are the JSON the model
            // wrote around the input); the whole input goes out now
            json added = output_item;
            added["status"] = "in_progress";
            added["input"]  = "";
            server_sent_events.push_back(json {
                {"event", "response.output_item.added"},
                {"data", json {
                    {"type", "response.output_item.added"},
                    {"output_index", item_output_index},
                    {"item", added}
                }}
            });
            server_sent_events.push_back(json {
                {"event", "response.custom_tool_call_input.delta"},
                {"data", json {
                    {"type",    "response.custom_tool_call_input.delta"},
                    {"item_id", output_item.at("id")},
                    {"output_index", item_output_index},
                    {"delta",   output_item.at("input")}
                }}
            });
            server_sent_events.push_back(json {
                {"event", "response.custom_tool_call_input.done"},
                {"data", json {
                    {"type",    "response.custom_tool_call_input.done"},
                    {"item_id", output_item.at("id")},
                    {"output_index", item_output_index},
                    {"input",   output_item.at("input")}
                }}
            });
        }
        server_sent_events.push_back(json {
            {"event", "response.output_item.done"},
            {"data", json {
                {"type", "response.output_item.done"},
                {"output_index", item_output_index},
                {"item", output_item}
            }}
        });
        output.push_back(output_item);
    }

    std::time_t t = std::time(0);
    server_sent_events.push_back(json {
        {"event", "response.completed"},
        {"data", json {
            {"type", "response.completed"},
            {"response", json {
                {"id",         oai_resp_id},
                {"object",     "response"},
                {"created_at", t},
                {"status",     "completed"},
                {"model",      oaicompat_model},
                {"output",     output},
                {"usage",      json {
                    {"input_tokens",  n_prompt_tokens},
                    {"output_tokens", n_decoded},
                    {"total_tokens",  n_decoded + n_prompt_tokens},
                    {"input_tokens_details", json { {"cached_tokens", n_prompt_tokens_cache} }},
                }}
            }},
        }}
    });

    if (stats.is_set()) {
        server_sent_events.back().at("data")["timings"] = stats.to_json();
    }

    return server_sent_events;
}

json server_task_result_cmpl_final::to_json_oaicompat_asr() {
    json event = json {
        {"type",  "transcript.text.done"},
        {"text",  oaicompat_msg.content},
        {"usage", json {
            {"type",         "tokens"},
            {"input_tokens",  n_prompt_tokens},
            {"output_tokens", n_decoded},
            {"total_tokens",  n_decoded + n_prompt_tokens},
            {"input_tokens_details", json { {"cached_tokens", n_prompt_tokens_cache} }},
        }},
    };
    return event;
}

json server_task_result_cmpl_final::to_json_anthropic() {
    std::string stop_reason = "max_tokens";
    if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
        stop_reason = oaicompat_msg.tool_calls.empty() ? "end_turn" : "tool_use";
    }

    json content_blocks = json::array();

    common_chat_msg msg;
    if (!oaicompat_msg.empty()) {
        msg = oaicompat_msg;
    } else {
        msg.role = "assistant";
        msg.content = content.text;
    }

    // thinking block comes first (Anthropic extended thinking format)
    if (!msg.reasoning_content.empty()) {
        content_blocks.push_back({
            {"type", "thinking"},
            {"thinking", msg.reasoning_content},
            {"signature", ""}  // empty signature for local models (no cryptographic verification)
        });
    }

    if (!msg.content.empty()) {
        content_blocks.push_back({
            {"type", "text"},
            {"text", msg.content}
        });
    }

    for (const auto & tool_call : msg.tool_calls) {
        json tool_use_block = {
            {"type", "tool_use"},
            {"id", tool_call.id},
            {"name", tool_call.name}
        };

        try {
            tool_use_block["input"] = json::parse(tool_call.arguments);
        } catch (const std::exception &) {
            tool_use_block["input"] = json::object();
        }

        content_blocks.push_back(tool_use_block);
    }

    json res = {
        {"id", oaicompat_cmpl_id},
        {"type", "message"},
        {"role", "assistant"},
        {"content", content_blocks},
        {"model", oaicompat_model},
        {"stop_reason", stop_reason},
        {"stop_sequence", stopping_word.empty() ? nullptr : json(stopping_word)},
        {"usage", {
            {"cache_read_input_tokens", n_prompt_tokens_cache},
            {"input_tokens", n_prompt_tokens - n_prompt_tokens_cache},
            {"output_tokens", n_decoded}
        }}
    };

    return res;
}

json server_task_result_cmpl_final::to_json_anthropic_stream() {
    json events = json::array();

    std::string stop_reason = "max_tokens";
    if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
        stop_reason = oaicompat_msg.tool_calls.empty() ? "end_turn" : "tool_use";
    }

    bool has_thinking = !oaicompat_msg.reasoning_content.empty();
    bool has_text     = !oaicompat_msg.content.empty();
    size_t num_tool_calls = oaicompat_msg.tool_calls.size();

    // content block indices: thinking (0) -> text (0 or 1) -> tool_use (n+)
    size_t thinking_block_index = 0;
    size_t text_block_index     = has_thinking ? 1 : 0;

    bool thinking_block_started = false;
    bool text_block_started     = false;
    std::unordered_set<size_t> tool_calls_started;

    for (const auto & diff : oaicompat_msg_diffs) {
        // handle thinking/reasoning content
        if (!diff.reasoning_content_delta.empty()) {
            if (!thinking_block_started) {
                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", thinking_block_index},
                        {"content_block", {
                            {"type", "thinking"},
                            {"thinking", ""}
                        }}
                    }}
                });
                thinking_block_started = true;
            }

            events.push_back({
                {"event", "content_block_delta"},
                {"data", {
                    {"type", "content_block_delta"},
                    {"index", thinking_block_index},
                    {"delta", {
                        {"type", "thinking_delta"},
                        {"thinking", diff.reasoning_content_delta}
                    }}
                }}
            });
        }

        // handle regular text content
        if (!diff.content_delta.empty()) {
            if (!text_block_started) {
                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", text_block_index},
                        {"content_block", {
                            {"type", "text"},
                            {"text", ""}
                        }}
                    }}
                });
                text_block_started = true;
            }

            events.push_back({
                {"event", "content_block_delta"},
                {"data", {
                    {"type", "content_block_delta"},
                    {"index", text_block_index},
                    {"delta", {
                        {"type", "text_delta"},
                        {"text", diff.content_delta}
                    }}
                }}
            });
        }

        // handle tool calls
        if (diff.tool_call_index != std::string::npos) {
            size_t content_block_index = (has_thinking ? 1 : 0) + (has_text ? 1 : 0) + diff.tool_call_index;

            if (tool_calls_started.find(diff.tool_call_index) == tool_calls_started.end()) {
                const auto & full_tool_call = oaicompat_msg.tool_calls[diff.tool_call_index];

                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", content_block_index},
                        {"content_block", {
                            {"type", "tool_use"},
                            {"id", full_tool_call.id},
                            {"name", full_tool_call.name}
                        }}
                    }}
                });
                tool_calls_started.insert(diff.tool_call_index);
            }

            if (!diff.tool_call_delta.arguments.empty()) {
                events.push_back({
                    {"event", "content_block_delta"},
                    {"data", {
                        {"type", "content_block_delta"},
                        {"index", content_block_index},
                        {"delta", {
                            {"type", "input_json_delta"},
                            {"partial_json", diff.tool_call_delta.arguments}
                        }}
                    }}
                });
            }
        }
    }

    // close content blocks in order
    if (has_thinking) {
        // Anthropic API requires a signature_delta before closing thinking blocks
        // We use an empty signature since we can't generate a cryptographic signature for local models
        events.push_back({
            {"event", "content_block_delta"},
            {"data", {
                {"type", "content_block_delta"},
                {"index", thinking_block_index},
                {"delta", {
                    {"type", "signature_delta"},
                    {"signature", ""}
                }}
            }}
        });
        events.push_back({
            {"event", "content_block_stop"},
            {"data", {
                {"type", "content_block_stop"},
                {"index", thinking_block_index}
            }}
        });
    }

    if (has_text) {
        events.push_back({
            {"event", "content_block_stop"},
            {"data", {
                {"type", "content_block_stop"},
                {"index", text_block_index}
            }}
        });
    }

    for (size_t i = 0; i < num_tool_calls; i++) {
        size_t content_block_index = (has_thinking ? 1 : 0) + (has_text ? 1 : 0) + i;
        events.push_back({
            {"event", "content_block_stop"},
            {"data", {
                {"type", "content_block_stop"},
                {"index", content_block_index}
            }}
        });
    }

    events.push_back({
        {"event", "message_delta"},
        {"data", {
            {"type", "message_delta"},
            {"delta", {
                {"stop_reason", stop_reason},
                {"stop_sequence", stopping_word.empty() ? nullptr : json(stopping_word)}
            }},
            {"usage", {
                {"output_tokens", n_decoded}
            }}
        }}
    });

    events.push_back({
        {"event", "message_stop"},
        {"data", {
            {"type", "message_stop"}
        }}
    });

    return events;
}

//
// server_task_result_cmpl_partial
//
void server_task_result_cmpl_partial::update(task_result_state & state) {
    is_updated = true;
    if (is_begin) {
        return; // begin marker only flushes headers, skip parsing
    }
    state.update_chat_msg(content, true, oaicompat_msg_diffs);

    // Copy current state for use in to_json_*() (reflects state BEFORE this chunk)
    thinking_block_started = state.thinking_block_started;
    text_block_started     = state.text_block_started;

    oai_resp_created       = state.oai_resp_created;
    oai_resp_id            = state.oai_resp_id;
    oai_resp_reasoning_id  = state.oai_resp_reasoning_id;
    oai_resp_message_id    = state.oai_resp_message_id;
    oai_resp_fc_id         = state.oai_resp_fc_id;
    oai_resp_fc_custom     = state.oai_resp_fc_custom;
    resp_tools             = state.resp_tools;

    // track if the accumulated message has any reasoning content
    anthropic_has_reasoning = !state.chat_msg.reasoning_content.empty();

    if (res_type == TASK_RESPONSE_TYPE_OAI_RESP && !state.oai_resp_created && (is_progress || n_decoded == 1)) {
        state.oai_resp_created = true;
    }

    // Pre-compute state updates based on diffs (for next chunk)
    for (const common_chat_msg_diff & diff : oaicompat_msg_diffs) {
        if (!diff.reasoning_content_delta.empty() && !state.thinking_block_started) {
            state.thinking_block_started = true;
        }
        if (!diff.content_delta.empty() && !state.text_block_started) {
            state.text_block_started = true;
        }
        if (!diff.tool_call_delta.name.empty()) {
            state.oai_resp_fc_id     = diff.tool_call_delta.id;
            state.oai_resp_fc_custom = resp_tool_is_custom(state.resp_tools, diff.tool_call_delta.name);
        }
    }
}

json server_task_result_cmpl_partial::to_json() {
    GGML_ASSERT(is_updated && "update() must be called before to_json()");
    if (is_begin) {
        return nullptr; // simply signal to HTTP handler to send the headers and status code
    }
    switch (res_type) {
        case TASK_RESPONSE_TYPE_NONE:
            return to_json_non_oaicompat();
        case TASK_RESPONSE_TYPE_OAI_CMPL:
            return to_json_oaicompat();
        case TASK_RESPONSE_TYPE_OAI_CHAT:
            return to_json_oaicompat_chat();
        case TASK_RESPONSE_TYPE_OAI_RESP:
            return to_json_oaicompat_resp();
        case TASK_RESPONSE_TYPE_OAI_ASR:
            return to_json_oaicompat_asr();
        case TASK_RESPONSE_TYPE_ANTHROPIC:
            return to_json_anthropic();
        default:
            GGML_ASSERT(false && "Invalid task_response_type");
    }
}

json server_task_result_cmpl_partial::to_json_non_oaicompat() {
    // non-OAI-compat JSON
    json res = json {
        {"index",            index},
        {"content",          content.text},
        {"tokens",           tokens},
        {"stop",             false},
        {"id_slot",          id_slot},
        {"tokens_predicted", n_decoded},
        {"tokens_evaluated", n_prompt_tokens},
    };
    // populate the timings object when needed (usually for the last response or with timings_per_token enabled)
    if (stats.is_set()) {
        res["timings"] = stats.to_json();
    }
    if (is_progress) {
        res["prompt_progress"] = progress.to_json();
    }
    if (!prob_output.probs.empty()) {
        res["completion_probabilities"] = completion_token_output::probs_vector_to_json({prob_output}, post_sampling_probs);
    }
    return res;
}

json server_task_result_cmpl_partial::to_json_oaicompat() {
    std::time_t t = std::time(0);
    json logprobs = json(nullptr); // OAI default to null
    if (prob_output.probs.size() > 0) {
        logprobs = json{
            {"content", completion_token_output::probs_vector_to_json({prob_output}, post_sampling_probs)},
        };
    }
    json res = json {
        {"choices",            json::array({
            json{
                {"text",          content.text},
                {"index",         index},
                {"logprobs",      logprobs},
                {"finish_reason", nullptr},
            }
        })},
        {"created",            t},
        {"model",              oaicompat_model},
        {"system_fingerprint", std::string(llama_build_info())},
        {"object",             "text_completion"},
        {"id",                 oaicompat_cmpl_id}
    };

    // extra fields for debugging purposes
    if (verbose) {
        res["__verbose"] = to_json_non_oaicompat();
    }
    if (stats.is_set()) {
        res["timings"] = stats.to_json();
    }
    if (is_progress) {
        res["prompt_progress"] = progress.to_json();
    }

    return res;
}

json server_task_result_cmpl_partial::to_json_oaicompat_chat() {
    bool first = n_decoded == 1;
    std::time_t t = std::time(0);
    json choices;

    std::vector<json> deltas;
    auto add_delta = [&](const json & delta) {
        deltas.push_back({
            {"choices", json::array({
                json {
                    {"finish_reason", nullptr},
                    {"index", index},
                    {"delta", delta},
                },
            })},
            {"created", t},
            {"id", oaicompat_cmpl_id},
            {"model", oaicompat_model},
            {"system_fingerprint", std::string(llama_build_info())},
            {"object", "chat.completion.chunk"},
        });
    };
    // We have to send an initial update to conform to openai behavior
    if (first || is_progress) {
        add_delta({
            {"role", "assistant"},
            {"content", nullptr},
        });
    }

    for (const auto & diff : oaicompat_msg_diffs) {
        add_delta(server_chat_msg_diff_to_json_oaicompat(diff));
    }

    if (!deltas.empty()) {
        auto & last_json = deltas[deltas.size() - 1];
        GGML_ASSERT(last_json.at("choices").size() >= 1);

        if (prob_output.probs.size() > 0) {
            last_json.at("choices").at(0)["logprobs"] = json {
                {"content", completion_token_output::probs_vector_to_json({prob_output}, post_sampling_probs)},
            };
        }

        if (stats.is_set()) {
            last_json["timings"] = stats.to_json();
        }
        if (is_progress) {
            last_json["prompt_progress"] = progress.to_json();
        }
    }

    return deltas;
}

json server_task_result_cmpl_partial::to_json_oaicompat_resp() {
    std::vector<json> events;

    if (!oai_resp_created) {
        events.push_back(json {
            {"event", "response.created"},
            {"data", json {
                {"type", "response.created"},
                {"response", json {
                    {"id",     oai_resp_id},
                    {"object", "response"},
                    {"status", "in_progress"},
                }},
            }},
        });
        events.push_back(json {
            {"event", "response.in_progress"},
            {"data", json {
                {"type", "response.in_progress"},
                {"response", json {
                    {"id",     oai_resp_id},
                    {"object", "response"},
                    {"status", "in_progress"},
                }},
            }},
        });
    } else if (is_progress) {
        events.push_back(json {
            {"event", "response.in_progress"},
            {"data", json {
                {"type", "response.in_progress"},
                {"response", json {
                    {"id",     oai_resp_id},
                    {"object", "response"},
                    {"status", "in_progress"},
                }},
            }},
        });
    }

    for (const common_chat_msg_diff & diff : oaicompat_msg_diffs) {
        if (!diff.reasoning_content_delta.empty()) {
            if (!thinking_block_started) {
                events.push_back(json {
                    {"event", "response.output_item.added"},
                    {"data", json {
                        {"type", "response.output_item.added"},
                        {"output_index", 0},
                        {"item", json {
                            {"id",                oai_resp_reasoning_id},
                            {"summary",           json::array()},
                            {"type",              "reasoning"},
                            {"content",           json::array()},
                            {"encrypted_content", ""},
                            {"status",            "in_progress"},
                        }},
                    }},
                });
                thinking_block_started = true;
            }
            events.push_back(json {
                {"event", "response.reasoning_text.delta"},
                {"data", json {
                    {"type",    "response.reasoning_text.delta"},
                    {"output_index", 0},
                    {"content_index", 0},
                    {"delta",   diff.reasoning_content_delta},
                    {"item_id", oai_resp_reasoning_id},
                }},
            });
        }

        if (!diff.content_delta.empty()) {
            if (!text_block_started) {
                events.push_back(json {
                    {"event", "response.output_item.added"},
                    {"data", json {
                        {"type", "response.output_item.added"},
                        {"output_index", anthropic_has_reasoning ? 1 : 0},
                        {"item", json {
                            {"content", json::array()},
                            {"id",      oai_resp_message_id},
                            {"role",    "assistant"},
                            {"status",  "in_progress"},
                            {"type",    "message"},
                        }},
                    }},
                });
                events.push_back(json {
                    {"event", "response.content_part.added"},
                    {"data", json {
                        {"type",    "response.content_part.added"},
                        {"item_id", oai_resp_message_id},
                        {"output_index", anthropic_has_reasoning ? 1 : 0},
                        {"content_index", 0},
                        {"part", json {
                            {"type", "output_text"},
                            {"text", ""},
                        }},
                    }},
                });
                text_block_started = true;
            }
            events.push_back(json {
                {"event", "response.output_text.delta"},
                {"data", json {
                    {"type",    "response.output_text.delta"},
                    {"item_id", oai_resp_message_id},
                    {"output_index", anthropic_has_reasoning ? 1 : 0},
                    {"content_index", 0},
                    {"delta",   diff.content_delta},
                }},
            });
        }

        if (!diff.tool_call_delta.name.empty()) {
            // strixllama: a custom tool's call goes out whole with the final result; a namespace tool's call names
            // the tool and its namespace (task_params::resp_tools)
            oai_resp_fc_custom = resp_tool_is_custom(resp_tools, diff.tool_call_delta.name);
            if (!oai_resp_fc_custom) {
                json item = json {
                    {"id",        "fc_" + diff.tool_call_delta.id},
                    {"arguments", ""},
                    {"call_id",   "call_" + diff.tool_call_delta.id},
                    {"name",      diff.tool_call_delta.name},
                    {"type",      "function_call"},
                    {"status",    "in_progress"},
                };
                if (const json * e = resp_tool_entry(resp_tools, diff.tool_call_delta.name)) {
                    item["name"] = json_value(*e, "name", diff.tool_call_delta.name);
                    if (e->contains("namespace")) {
                        item["namespace"] = e->at("namespace");
                    }
                }
                events.push_back(json {
                    {"event", "response.output_item.added"},
                    {"data", json {
                        {"type",  "response.output_item.added"},
                        {"output_index", (anthropic_has_reasoning ? 1 : 0) + (text_block_started ? 1 : 0) + diff.tool_call_index},
                        {"item", item},
                    }},
                });
            }
            oai_resp_fc_id = diff.tool_call_delta.id;
        }

        if (!diff.tool_call_delta.arguments.empty() && !oai_resp_fc_custom) {
            events.push_back(json {
                {"event", "response.function_call_arguments.delta"},
                {"data", json {
                    {"type",    "response.function_call_arguments.delta"},
                    {"delta",   diff.tool_call_delta.arguments},
                    {"item_id", "fc_" + oai_resp_fc_id},
                    {"output_index", (anthropic_has_reasoning ? 1 : 0) + (text_block_started ? 1 : 0) + diff.tool_call_index},
                }},
            });
        }
    }

    if (!events.empty()) {
        json & data = events.back().at("data");
        if (stats.is_set()) {
            data["timings"] = stats.to_json();
        }
        if (is_progress) {
            data["prompt_progress"] = progress.to_json();
        }
    }

    return events;
}

json server_task_result_cmpl_partial::to_json_oaicompat_asr() {
    json event = json {
        {"type", "transcript.text.delta"},
        {"delta", content.text},
    };
    return event;
}

json server_task_result_cmpl_partial::to_json_anthropic() {
    json events = json::array();
    bool first = (n_decoded == 1);
    // use member variables to track block state across streaming calls
    // (anthropic_thinking_block_started, anthropic_text_block_started)

    if (first) {
        events.push_back({
            {"event", "message_start"},
            {"data", {
                {"type", "message_start"},
                {"message", {
                    {"id", oaicompat_cmpl_id},
                    {"type", "message"},
                    {"role", "assistant"},
                    {"content", json::array()},
                    {"model", oaicompat_model},
                    {"stop_reason", nullptr},
                    {"stop_sequence", nullptr},
                    {"usage", {
                        {"cache_read_input_tokens", n_prompt_tokens_cache},
                        {"input_tokens", n_prompt_tokens - n_prompt_tokens_cache},
                        {"output_tokens", 0}
                    }}
                }}
            }}
        });
    }

    // content block indices: thinking (0) -> text (0 or 1) -> tool_use (n+)
    size_t thinking_block_index = 0;
    // use anthropic_has_reasoning (set in update()) to know if ANY reasoning was generated
    size_t text_block_index     = anthropic_has_reasoning ? 1 : 0;

    // use local copies of streaming state (copied from task_result_state in update())
    // these reflect the state BEFORE this chunk was processed
    bool thinking_started = thinking_block_started;
    bool text_started     = text_block_started;

    for (const auto & diff : oaicompat_msg_diffs) {
        // handle thinking/reasoning content
        if (!diff.reasoning_content_delta.empty()) {
            if (!thinking_started) {
                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", thinking_block_index},
                        {"content_block", {
                            {"type", "thinking"},
                            {"thinking", ""}
                        }}
                    }}
                });
                thinking_started = true;
            }

            events.push_back({
                {"event", "content_block_delta"},
                {"data", {
                    {"type", "content_block_delta"},
                    {"index", thinking_block_index},
                    {"delta", {
                        {"type", "thinking_delta"},
                        {"thinking", diff.reasoning_content_delta}
                    }}
                }}
            });
        }

        // handle regular text content
        if (!diff.content_delta.empty()) {
            if (!text_started) {
                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", text_block_index},
                        {"content_block", {
                            {"type", "text"},
                            {"text", ""}
                        }}
                    }}
                });
                text_started = true;
            }

            events.push_back({
                {"event", "content_block_delta"},
                {"data", {
                    {"type", "content_block_delta"},
                    {"index", text_block_index},
                    {"delta", {
                        {"type", "text_delta"},
                        {"text", diff.content_delta}
                    }}
                }}
            });
        }

        // handle tool calls
        if (diff.tool_call_index != std::string::npos) {
            // use anthropic_has_reasoning for thinking block count (persists across calls)
            size_t content_block_index = (anthropic_has_reasoning ? 1 : 0) + (text_started ? 1 : 0) + diff.tool_call_index;

            if (!diff.tool_call_delta.name.empty()) {
                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", content_block_index},
                        {"content_block", {
                            {"type", "tool_use"},
                            {"id", diff.tool_call_delta.id},
                            {"name", diff.tool_call_delta.name}
                        }}
                    }}
                });
            }

            if (!diff.tool_call_delta.arguments.empty()) {
                events.push_back({
                    {"event", "content_block_delta"},
                    {"data", {
                        {"type", "content_block_delta"},
                        {"index", content_block_index},
                        {"delta", {
                            {"type", "input_json_delta"},
                            {"partial_json", diff.tool_call_delta.arguments}
                        }}
                    }}
                });
            }
        }
    }

    return events;
}

//
// server_task_result_embd
//
json server_task_result_embd::to_json() {
    return res_type == TASK_RESPONSE_TYPE_OAI_EMBD
        ? to_json_oaicompat()
        : to_json_non_oaicompat();
}

json server_task_result_embd::to_json_non_oaicompat() {
    return json {
        {"index",     index},
        {"embedding", embedding},
    };
}

json server_task_result_embd::to_json_oaicompat() {
    return json {
        {"index",            index},
        {"embedding",        embedding[0]},
        {"tokens_evaluated", n_tokens},
    };
}

//
// server_task_result_rerank
//
json server_task_result_rerank::to_json() {
    return json {
        {"index",            index},
        {"score",            score},
        {"tokens_evaluated", n_tokens},
    };
}

//
// server_task_result_tts
//
json server_task_result_tts::to_json() {
    return json {
        {"sample_rate", sample_rate},
        {"n_bytes",     audio.size()},
        {"final",       final},
    };
}

//
// server_task_result_decision
//
json server_task_result_decision::to_json() {
    return json {
        {"index",            index},
        {"scores",           scores},
        {"tokens_evaluated", n_tokens},
    };
}

//
// server_task_result_error
//
json server_task_result_error::to_json() {
    json res = format_error_response(err_msg, err_type);
    if (err_type == ERROR_TYPE_EXCEED_CONTEXT_SIZE) {
        res["n_prompt_tokens"] = n_prompt_tokens;
        res["n_ctx"]           = n_ctx;
    }
    return res;
}

//
// server_task_result_metrics
//
json server_task_result_slots::to_json() {
    return slots_data;
}

json server_task_result_metrics::to_json() {
    // not used, /metrics renders prometheus text via to_metrics()
    return json{};
}

// metrics definition: https://prometheus.io/docs/practices/naming/#metric-names
std::string server_task_result_metrics::to_metrics() {
    const std::vector<metric_item> counters = {
        {
            "prompt_tokens_total",
            "Number of prompt tokens processed, excluding cached tokens",
            (double) metrics.prompt.count
        }, {
            "prompt_tokens_cached_total",
            "Number of prompt tokens reused from the cache",
            (double) metrics.n_prompt_cached
        }, {
            "prompt_seconds_total",
            "Total time spent processing prompts",
            metrics.prompt.time / 1.e6
        }, {
            "tokens_predicted_total",
            "Number of generation tokens processed",
            (double) metrics.predict.count
        }, {
            "tokens_predicted_seconds_total",
            "Total time spent generating tokens",
            metrics.predict.time / 1.e6
        }, {
            "n_decode_total",
            "Total number of llama_decode() calls, excluding speculative decoding and multimodal decoding",
            (double) metrics.n_decode
        }, {
            "n_tokens_max",
            "Largest observed sequence length (prompt + generation)",
            (double) metrics.n_tokens_max
        }, {
            "spec_decode_num_draft_tokens_total",
            "Speculative: Total draft tokens generated",
            (double) metrics.n_draft_tokens
        }, {
            "spec_decode_num_accepted_tokens_total",
            "Speculative: Total draft tokens accepted by the target model",
            (double) metrics.n_draft_accepted
        }, {
            "spec_decode_num_drafts_total",
            "Speculative: Total speculative decoding verification steps",
            (double) metrics.n_draft_verif_steps
        },
    };

    const std::vector<metric_item> gauges = {
        {
            "prompt_tokens_seconds",
            "Average prompt throughput in tokens/s",
            metrics.prompt_bucket.n_per_second()
        }, {
            "predicted_tokens_seconds",
            "Average generation throughput in tokens/s",
            metrics.predict_bucket.n_per_second()
        }, {
            "requests_processing",
            "Number of requests processing",
            (double) n_processing_slots
        }, {
            "requests_deferred",
            "Number of requests deferred",
            (double) n_tasks_deferred
        }, {
            "n_busy_slots_per_decode",
            "Average number of busy slots per llama_decode() call",
            (double) metrics.n_busy_slots / std::max((double) metrics.n_decode, 1.0)
        },
    };

    std::stringstream prometheus;

    auto add_items = [&prometheus](const char * type, const std::vector<metric_item> & items) {
        for (const auto & item : items) {
            prometheus << "# HELP llamacpp:" << item.name << " " << item.description << "\n"
                       << "# TYPE llamacpp:" << item.name << " " << type             << "\n"
                       << "llamacpp:"        << item.name << " " << item.value       << "\n";
        }
    };

    add_items("counter", counters);
    add_items("gauge",   gauges);

    // labeled counter: one time series per draft position
    if (!metrics.n_accepted_per_pos.empty()) {
        prometheus << "# HELP llamacpp:spec_decode_num_accepted_tokens_per_pos_total"
                      " Accepted tokens per draft position\n"
                   << "# TYPE llamacpp:spec_decode_num_accepted_tokens_per_pos_total counter\n";
        for (size_t i = 0; i < metrics.n_accepted_per_pos.size(); i++) {
            prometheus << "llamacpp:spec_decode_num_accepted_tokens_per_pos_total{position=\""
                       << i << "\"} " << metrics.n_accepted_per_pos[i] << "\n";
        }
    }

    return prometheus.str();
}

//
// server_task_result_slot_save_load
//
json server_task_result_slot_save_load::to_json() {
    if (is_save) {
        return json {
            { "id_slot",   id_slot },
            { "filename",  filename },
            { "n_saved",   n_tokens },
            { "n_written", n_bytes },
            { "timings", {
                { "save_ms", t_ms }
            }},
        };
    }

    return json {
        { "id_slot",    id_slot },
        { "filename",   filename },
        { "n_restored", n_tokens },
        { "n_read",     n_bytes },
        { "timings", {
            { "restore_ms", t_ms }
        }},
    };
}

//
// server_task_result_slot_erase
//
json server_task_result_slot_erase::to_json() {
    return json {
        { "id_slot",  id_slot },
        { "n_erased", n_erased },
    };
}

//
// server_task_result_get_lora
//

json server_task_result_get_lora::to_json() {
    json result = json::array();
    for (size_t i = 0; i < loras.size(); ++i) {
        auto & lora = loras[i];
        json entry = {
            {"id",            i},
            {"path",          lora.info.path},
            {"scale",         lora.info.scale},
            {"task_name",     lora.info.task_name},
            {"prompt_prefix", lora.info.prompt_prefix},
        };
        if (!lora.alora_invocation_tokens.empty()) {
            entry["alora_invocation_string"] = lora.alora_invocation_string;
            entry["alora_invocation_tokens"] = lora.alora_invocation_tokens;
        }
        result.push_back(std::move(entry));
    }
    return result;
}

//
// server_task_result_apply_lora
//

json server_task_result_apply_lora::to_json() {
    return json {{ "success", true }};
}

//
// server_prompt_cache
//
size_t server_prompt_cache::size() const {
    size_t res = 0;

    for (const auto & state : states) {
        res += state.size();
    }

    return res;
}

size_t server_prompt_cache::n_tokens() const {
    size_t res = 0;

    for (const auto & state : states) {
        res += state.prompt.n_tokens();
    }

    return res;
}

server_prompt_cache_state * server_prompt_cache::alloc(const server_prompt & prompt, size_t state_size_tgt, size_t state_size_dft) {
    // first check if the current state is contained fully in the cache
    for (auto it = states.begin(); it != states.end(); ++it) {
        const int cur_lcp_len = it->prompt.tokens.get_common_prefix(prompt.tokens);

        if (cur_lcp_len == (int) prompt.tokens.size()) {
            SRV_TRC("%s", " - prompt is already in the cache, skipping\n");
            return nullptr;
        }
    }

    // calculate checkpoints size to see if it will fit with the prompt
    size_t checkpoints_size = 0;
    for (const auto & ckpt : prompt.checkpoints) {
        checkpoints_size += ckpt.size();
    }

    const size_t state_size_new = state_size_tgt + state_size_dft + checkpoints_size;

    // skip over-limit entries to avoid disturbing the cache
    if (limit_size > 0 && state_size_new > limit_size) {
        SRV_WRN(" - prompt state size %.3f MiB exceeds cache size limit %.3f MiB, skipping\n",
                state_size_new / (1024.0 * 1024.0), limit_size / (1024.0 * 1024.0));
        return nullptr;
    }

    // remove any cached prompts that are fully contained in the current prompt
    for (auto it = states.begin(); it != states.end();) {
        const int len = it->prompt.tokens.get_common_prefix(prompt.tokens);

        if (len == (int) it->prompt.tokens.size()) {
            SRV_TRC(" - removing obsolete cached prompt with length %d\n", len);

            it = states.erase(it);
        } else {
            ++it;
        }
    }

    if (limit_size > 0) {
        // make room before allocating the new vectors to avoid breaching the limit
        while (!states.empty() && size() + state_size_new > limit_size) {
            SRV_WRN(" - making room for prompt cache entry, removing oldest entry (size = %.3f MiB)\n",
                    states.front().size() / (1024.0 * 1024.0));

            states.pop_front();
        }
    }

    std::vector<uint8_t> state_data_tgt;
    std::vector<uint8_t> state_data_dft;

    // check if we can allocate enough memory for the new state
    try {
        state_data_tgt.resize(state_size_tgt);
        state_data_dft.resize(state_size_dft);
    } catch (const std::bad_alloc & e) {
        SRV_ERR("failed to allocate memory for prompt cache state: %s\n", e.what());

        limit_size = std::max<size_t>(1, 0.4*size());

        SRV_WRN(" - cache size limit reduced to %.3f MiB\n", limit_size / (1024.0 * 1024.0));

        update();

        return nullptr;
    }

    states.push_back({
        /*.prompt =*/ {
            /*.tokens      =*/ prompt.tokens.clone(),
            /*.checkpoints =*/ prompt.checkpoints,
        },
        /*.data   =*/ {
            /*.main =*/ std::move(state_data_tgt),
            /*.drft =*/ std::move(state_data_dft),
            /*.spec =*/ {},
        },
    });

    return &states.back();
}

bool server_prompt_cache::load(server_prompt & prompt, const server_tokens & tokens_new, llama_context * ctx_tgt, llama_context * ctx_dft, int32_t id_slot,
                               const std::function<void(size_t)> & before_restore, ckpt_paged_map * paged_out, disk_runs_state * runs_out,
                               std::vector<uint8_t> * spec_out) {
    if (spec_out) {
        spec_out->clear();
    }
    const int lcp_best = prompt.tokens.get_common_prefix(tokens_new);

    float f_keep_best = prompt.tokens.size() > 0 ? float(lcp_best) / prompt.tokens.size() : -1.0f; // empty slot: any cache entry wins
    float f_sim_best  = float(lcp_best) / tokens_new.size();

    SRV_TRC(" - looking for better prompt, base f_keep = %.3f, f_sim = %.3f\n", f_keep_best, f_sim_best);

    auto it_best = states.end();

    // find the most similar cached prompt, that would also preserve the most context
    for (auto it = states.begin(); it != states.end(); ++it) {
        const int lcp_cur = it->prompt.tokens.get_common_prefix(tokens_new);

        const float f_keep_cur = float(lcp_cur) / it->prompt.tokens.size();
        const float f_sim_cur  = float(lcp_cur) / tokens_new.size();

        SRV_TRC("   - prompt with length %7zu, lcp = %7d, f_keep = %.3f, f_sim = %.3f\n", it->prompt.tokens.size(), lcp_cur, f_keep_cur, f_sim_cur);

        // don't trash large prompts
        if (f_keep_cur < 0.25f) {
            continue;
        }

        if (f_keep_best < f_keep_cur && f_sim_best < f_sim_cur) {
            f_keep_best = f_keep_cur;
            f_sim_best  = f_sim_cur;

            it_best = it;
        }
    }

    // strixllama: a version 3 entry of the disk tier, when it beats what is in RAM, goes straight into the cache
    if (disk_limit > 0) {
        bool restored = false;
        if (load_runs(prompt, tokens_new, ctx_tgt, ctx_dft, id_slot, f_keep_best, f_sim_best, before_restore, paged_out,
                      runs_out, restored, spec_out)) {
            return restored;                       // attempted: whatever happened, the slot's own conversation went
        }
    }

    // strixllama: the disk tier, with the same criteria, only when it would beat what is in RAM. Its checkpoints
    // stay on disk when the caller can page them in (disk_paged holds their pins until the slot takes them); such
    // a state must never stay in the RAM tier, whose entries are complete, so every way out below drops it
    ckpt_paged_map disk_paged;
    bool from_disk = false;
    if (disk_limit > 0) {
        if (server_prompt_cache_state * s = load_from_disk(tokens_new, f_keep_best, f_sim_best, paged_out ? &disk_paged : nullptr)) {
            it_best = std::prev(states.end());
            GGML_ASSERT(&*it_best == s);
            from_disk = true;
        }
    }
    auto drop_from_disk = [&]() {
        if (from_disk) {
            states.erase(it_best);
            it_best = states.end();
            ckpt_release(disk_paged);
            from_disk = false;
        }
    };

    // strixllama: an entry written while MTP was off carries no draft state, and nothing here can
    // build one for a sequence the target is already deep into. Restoring the target alone would
    // leave the draft context empty and drafting from it would not match. Cheaper to be wrong about
    // the cache than about the tokens: process the prompt instead.
    if (it_best != states.end() && ctx_dft && it_best->data.drft.empty()) {
        SRV_WRN("%s", " - cached prompt carries no draft state and this server drafts: processing the prompt instead\n");
        if (from_disk) {
            drop_from_disk();
        } else {
            states.erase(it_best);
            it_best = states.end();
        }
    }

    if (it_best != states.end()) {
        SRV_TRC(" - found better prompt with f_keep = %.3f, f_sim = %.3f\n", f_keep_best, f_sim_best);

        if (before_restore) {
            before_restore(it_best->prompt.tokens.size());
        }
        // a whole state replaces the slot's conversation: the runs it was writing are not this one's
        if (runs_out) {
            runs_out->tgt.clear();
            runs_out->dft.clear();
            runs_out->tokens.clear();
        }

        {
            auto & data = it_best->data.main;

            const size_t size = data.size();
            const size_t n = llama_state_seq_set_data_ext(ctx_tgt, data.data(), size, id_slot, 0);
            if (n != size) {
                SRV_ERR("failed to restore state with size %zu\n", size);
                drop_from_disk();

                return false;
            }

            data.clear();
            data.shrink_to_fit();
        }

        {
            auto & data = it_best->data.drft;

            // strixllama: this server may have MTP off while the entry was written with it on. There is
            // then no draft context to restore into and no drafting either, so the target state is
            // complete on its own - drop the draft half rather than assert on it, which is what aborted
            // the server the first time a cached conversation was reopened with MTP turned off.
            if (!data.empty() && !ctx_dft) {
                SRV_INF("%s", " - cached prompt carries a draft state and this server does not draft: keeping the target state\n");
                data.clear();
                data.shrink_to_fit();
            }

            if (!data.empty()) {
                const size_t size = data.size();
                const size_t n = llama_state_seq_set_data_ext(ctx_dft, data.data(), size, id_slot, 0);
                if (n != size) {
                    SRV_WRN("failed to restore state with size %zu\n", size);
                    drop_from_disk();

                    return false;
                }

                data.clear();
                data.shrink_to_fit();
            }
        }

        prompt = std::move(it_best->prompt);
        if (spec_out) {
            *spec_out = std::move(it_best->data.spec);
        }

        states.erase(it_best);

        // strixllama: the slot takes the pins of the checkpoints that stayed on disk; a RAM-tier state brings its
        // checkpoints whole
        if (paged_out) {
            *paged_out = from_disk ? std::move(disk_paged) : ckpt_paged_map {};
        }
        from_disk = false;
    }

    return true;
}

void server_prompt_cache::update() {
    if (limit_size > 0) {
        while (!states.empty() && size() > limit_size) {
            SRV_WRN(" - cache size limit reached, removing oldest entry (size = %.3f MiB)\n", states.front().size() / (1024.0 * 1024.0));

            states.pop_front();
        }
    }

    // average size per token
    const float size_per_token = std::max<float>(1.0f, float(size()) / (std::max<size_t>(1, n_tokens())));

    // dynamically increase the token limit if it can fit in the memory limit
    const size_t limit_tokens_cur = limit_size > 0 ? std::max<size_t>(limit_tokens, limit_size/size_per_token) : limit_tokens;

    if (limit_tokens > 0) {
        while (!states.empty() && n_tokens() > limit_tokens_cur) {
            SRV_WRN(" - cache token limit (%zu, est: %zu) reached, removing oldest entry (size = %.3f MiB)\n",
                    limit_tokens, limit_tokens_cur, states.front().size() / (1024.0 * 1024.0));

            states.pop_front();
        }
    }

    SRV_TRC(" - cache state: %zu prompts, %.3f MiB (limits: %.3f MiB, %zu tokens, %zu est)\n",
            states.size(), size() / (1024.0 * 1024.0), limit_size / (1024.0 * 1024.0), limit_tokens, limit_tokens_cur);

    for (const auto & state : states) {
        SRV_TRC("   - prompt %p: %7d tokens, checkpoints: %2zu, %9.3f MiB\n",
                (const void *)&state, state.prompt.n_tokens(), state.prompt.checkpoints.size(), state.size() / (1024.0 * 1024.0));
    }
}

// ================================================================================================
// strixllama: the disk tier (declared in server-task.h)
// ================================================================================================
//
// Version 2 keeps an entry as a small manifest plus objects shared between entries, so saving a
// conversation again writes only what changed instead of several GB at once:
//
//   <xxh64>-<n>.spc      manifest: tokens, the checkpoints it uses, the chunks of its two states
//   ckpt/<key>.ckp       one context checkpoint. Immutable once made, and keyed by the token prefix it
//                        covers, so every later save of the same conversation reuses it
//   chunks/xx/<h>.chk    a piece of a serialised state, cut where its content says (content-defined
//                        chunking) and named by its XXH3-128. A KV cache grows by rows appended to every
//                        layer's section, so the old bytes survive whole, only further along; cuts made by
//                        content find them again wherever they land. Measured on a 79K-token conversation
//                        saved twice: 23 of 24 sampled windows of the old target state were in the new one,
//                        the missing one in the recurrent part, which changes on every save.
//
// One background thread writes everything and is the only thing that deletes: the main loop gathers the
// state out of the KV cache (~0.1 s for 2.3 GiB) and hands it over. Files appear as .part and are renamed
// once complete, and the manifest goes last, so a process killed mid-write leaves nothing a later start
// would trust - stray .part files and objects no manifest references are swept at startup.
//
// A store holds the format this build writes for the model, and nothing older: entries of an earlier format
// are deleted at startup, not read and not converted - they may be from any build back to 0.1.4, and the
// conversations in them are processed again once. That is version 3 for a model whose contexts serve rows
// (llama_strix_kv_row_size), version 2 for any other. A format newer than this build knows is left alone.
//
// manifest, version 2:
//   "STRIXSPC" u32 version u32 has_mtmd
//   u64 n_token_bytes, bytes                          (server_tokens::serialize)
//   u32 n_ckpt, per checkpoint:
//       u64 key i64 n_tokens i32 pos_min i32 pos_max u64 size_tgt u64 size_dft u64 size_spec
//   u64 main_size u32 n_chunks, per chunk: u64 xxh_high u64 xxh_low u32 size
//   u64 drft_size u32 n_chunks, the same
// checkpoint file, version 2:
//   "STRIXCKP" u32 version i64 n_tokens i32 pos_min i32 pos_max, then tgt, dft, spec as u64 n + bytes,
//   then u64 xxh_high u64 xxh_low: XXH3-128 of the three payloads
//
// Version 3 does not serialise the state at all. Version 2 still gathered a conversation's whole state into
// one buffer on every save (5 GiB at 173K tokens) and read it back whole, and appending to a KV cache moves
// bytes in every layer's section, so each save rewrote ~200 chunks. Version 3 keeps the attention rows by
// position (llama_strix_kv_*), in runs of up to disk_run positions that are written once, when a run fills
// up or the conversation leaves its slot, and never again - the next run starts where the last one ends.
// The recurrent state is a checkpoint, as above, and the part that changes: it is written only as the
// conversation leaves its slot (the state at the end and its last prompt's latest checkpoints), so a store
// entry for a conversation still in a slot restores only as far as it did the last time it left one.
// A restore takes the latest checkpoint the runs reach:
// cells for its positions, the rows a run at a time, then the recurrent state; the rest of the prompt is
// processed as any other.
//   runs/xx/<h>.run      "STRIXRUN" u32 version u64 row_size u32 n, then n rows, then u64 xxh_high u64
//                        xxh_low: XXH3-128 of the rows, which is also the name
// manifest, version 3: version 2's up to the checkpoints, then
//   u64 row_tgt u32 n_runs, per run: u64 xxh_high u64 xxh_low i32 pos0 i32 n i32 n_use
//   u64 row_dft u32 n_runs, the same
// (version 1, 0.1.4 - 0.1.6, was one self-contained file per conversation)
namespace {

namespace fs = std::filesystem;
using disk_entry     = server_prompt_cache::disk_entry;
using disk_chunk_ref = server_prompt_cache::disk_chunk_ref;
using disk_ckpt_ref  = server_prompt_cache::disk_ckpt_ref;
using disk_run_ref   = server_prompt_cache::disk_run_ref;

const char     SPC_MAGIC[8] = {'S','T','R','I','X','S','P','C'};
const char     CKP_MAGIC[8] = {'S','T','R','I','X','C','K','P'};
const char     RUN_MAGIC[8] = {'S','T','R','I','X','R','U','N'};
const uint32_t SPC_V2 = 2;
// the version 3 layout, numbered 4 since 0.2.8 so that a store drops what earlier builds wrote: 0.2.7 could compute a
// prompt chunk's attention densely over several conversations' cells (qwen4exp_qsa_maskless_ok), and its runs and
// checkpoints of such a conversation carry that on
const uint32_t SPC_V3 = 4;
const uint32_t CKP_V2 = 2;             // version 1 never shipped
const uint32_t RUN_V1 = 1;

// --- content-defined chunking ------------------------------------------------------------------------
// A cut where the gear hash of the last 64 bytes has its top 21 bits clear, no sooner than CDC_MIN after
// the previous cut and no later than CDC_MAX: ~2.5 MiB on average. The cut depends only on the bytes, so
// an unchanged stretch of a state is cut the same way however far an insertion before it moved it.
constexpr size_t   CDC_MIN  = 512u << 10;
constexpr size_t   CDC_MAX  = 8u << 20;
constexpr uint64_t CDC_MASK = ~0ull << (64 - 21);

const uint64_t * cdc_gear() {
    static const std::array<uint64_t, 256> gear = [] {
        std::array<uint64_t, 256> g {};
        uint64_t x = 0x5354524958434443ull;        // fixed: cuts must land in the same places in every run
        for (auto & v : g) {                        // splitmix64
            x += 0x9E3779B97F4A7C15ull;
            uint64_t z = x;
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
            v = z ^ (z >> 31);
        }
        return g;
    }();
    return gear.data();
}

// the length of the chunk that starts at p, with n bytes of the blob left from there
size_t cdc_next(const uint8_t * p, size_t n) {
    if (n <= CDC_MIN) {
        return n;
    }
    const uint64_t * G = cdc_gear();
    const size_t limit = std::min(n, CDC_MAX);
    uint64_t h = 0;
    for (size_t i = CDC_MIN - 64; i < limit; ++i) {
        h = (h << 1) + G[p[i]];
        if (i >= CDC_MIN && (h & CDC_MASK) == 0) {
            return i + 1;
        }
    }
    return limit;
}

disk_chunk_ref chunk_ref(const uint8_t * p, size_t n) {
    const XXH128_hash_t h = XXH3_128bits(p, n);
    return { h.high64, h.low64, (uint32_t) n };
}

std::string chunk_hex(uint64_t hi, uint64_t lo) {
    char b[33];
    snprintf(b, sizeof(b), "%016llx%016llx", (unsigned long long) hi, (unsigned long long) lo);
    return b;
}

std::string chunk_path(const std::string & dir, uint64_t hi, uint64_t lo) {
    const std::string hex = chunk_hex(hi, lo);
    return (fs::path(dir) / "chunks" / hex.substr(0, 2) / (hex + ".chk")).string();
}

std::string run_path(const std::string & dir, uint64_t hi, uint64_t lo) {
    const std::string hex = chunk_hex(hi, lo);
    return (fs::path(dir) / "runs" / hex.substr(0, 2) / (hex + ".run")).string();
}

uint64_t run_file_bytes(uint64_t row, int32_t n) {
    return 8 + sizeof(uint32_t) + sizeof(uint64_t) + sizeof(uint32_t) + row * (uint64_t) n + 2*sizeof(uint64_t);
}

int64_t runs_cover(const std::vector<disk_run_ref> & runs) {
    int64_t n = 0;
    for (const auto & r : runs) {
        n += r.n_use;
    }
    return n;
}

// the tokens a version 3 entry brings back exactly: its latest checkpoint that the runs reach
int64_t runs_exact(const server_tokens & tokens, const std::vector<disk_run_ref> & tgt, const std::vector<disk_run_ref> & dft,
                   bool has_dft, const std::vector<disk_ckpt_ref> & ckpts) {
    int64_t cover = std::min<int64_t>(runs_cover(tgt), (int64_t) tokens.size());
    if (has_dft) {
        cover = std::min(cover, runs_cover(dft));
    }
    int64_t best = 0;
    for (const auto & k : ckpts) {
        if (k.n_tokens <= cover) {
            best = std::max(best, k.n_tokens);
        }
    }
    return best;
}

std::string ckpt_path(const std::string & dir, uint64_t key) {
    char b[32];
    snprintf(b, sizeof(b), "%016llx.ckp", (unsigned long long) key);
    return (fs::path(dir) / "ckpt" / b).string();
}

// a checkpoint is the recurrent state after the first n_tokens of a conversation, so the tokens it covers
// name it; the rest only guards against two different things claiming one name
uint64_t ckpt_key(const llama_tokens & text, int64_t n_tokens, int32_t pos_min, int32_t pos_max,
                  uint64_t size_tgt, uint64_t size_dft, uint64_t size_spec) {
    const size_t n = (size_t) std::clamp<int64_t>(n_tokens, 0, (int64_t) text.size());
    const uint64_t seed = XXH3_64bits(text.data(), n * sizeof(llama_token));
    const uint64_t meta[6] = { (uint64_t) n_tokens, (uint64_t) (uint32_t) pos_min, (uint64_t) (uint32_t) pos_max,
                               size_tgt, size_dft, size_spec };
    return XXH3_64bits_withSeed(meta, sizeof(meta), seed);
}

// --- files ------------------------------------------------------------------------------------------
// Entries are read once and handed to the GPU, and written once and not read again until a conversation
// returns, so the page cache only costs here: it copies every GiB twice and evicts what the desktop was
// using (measured: 505 MB/s read through it against 3451 MB/s from the device). On Windows files are opened
// FILE_FLAG_NO_BUFFERING and moved through one aligned staging buffer that the caller owns, so a load that
// opens a thousand chunk files allocates it once.
struct spc_stage {
    static constexpr size_t SIZE = 8u << 20;        // >= CDC_MAX, and a multiple of every sector size
    uint8_t * p = nullptr;
    spc_stage() {
#ifdef _WIN32
        p = (uint8_t *) VirtualAlloc(nullptr, SIZE, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
#else
        p = (uint8_t *) std::aligned_alloc(4096, SIZE);
#endif
        if (!p) {
            throw std::bad_alloc();
        }
    }
    ~spc_stage() {
#ifdef _WIN32
        VirtualFree(p, 0, MEM_RELEASE);
#else
        std::free(p);
#endif
    }
    spc_stage(const spc_stage &) = delete;
    spc_stage & operator=(const spc_stage &) = delete;
};

struct spc_file {
    explicit spc_file(spc_stage & s) : st(s) {}
    ~spc_file() { close(); }
    spc_file(const spc_file &) = delete;
    spc_file & operator=(const spc_file &) = delete;

    bool open(const std::string & path) {
        close();
        have = used = 0;
        bad = false;
#ifdef _WIN32
        const std::wstring w = fs::path(path).wstring();
        h = CreateFileW(w.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                        FILE_FLAG_NO_BUFFERING | FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
        if (h == INVALID_HANDLE_VALUE) {
            // not every volume serves unbuffered reads; a buffered handle is slower, not broken
            h = CreateFileW(w.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                            FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
        }
        return h != INVALID_HANDLE_VALUE;
#else
        f.open(path, std::ios::binary);
        return (bool) f;
#endif
    }

    void close() {
#ifdef _WIN32
        if (h != INVALID_HANDLE_VALUE) {
            CloseHandle(h);
            h = INVALID_HANDLE_VALUE;
        }
#else
        if (f.is_open()) {
            f.close();
        }
#endif
    }

    // sequential; dst may be null to skip
    bool read(void * dst, size_t n) {
        uint8_t * out = (uint8_t *) dst;
        while (n > 0) {
            if (used == have && !refill()) {
                return false;
            }
            const size_t take = std::min(n, have - used);
            if (out) {
                memcpy(out, st.p + used, take);
                out += take;
            }
            used += take;
            n    -= take;
        }
        return true;
    }

private:
    spc_stage & st;
    size_t have = 0;
    size_t used = 0;
    bool   bad  = false;
#ifdef _WIN32
    HANDLE h = INVALID_HANDLE_VALUE;
#else
    std::ifstream f;
#endif

    bool refill() {
        if (bad) {
            return false;
        }
#ifdef _WIN32
        DWORD got = 0;
        if (h == INVALID_HANDLE_VALUE || !ReadFile(h, st.p, (DWORD) spc_stage::SIZE, &got, nullptr)) {
            bad = true;
            return false;
        }
        have = got;
#else
        if (!f.is_open()) {
            bad = true;
            return false;
        }
        f.read((char *) st.p, (std::streamsize) spc_stage::SIZE);
        have = (size_t) f.gcount();
#endif
        used = 0;
        return have > 0;
    }
};

struct spc_out {
    explicit spc_out(spc_stage & s) : st(s) {}
    ~spc_out() { abort(); }
    spc_out(const spc_out &) = delete;
    spc_out & operator=(const spc_out &) = delete;

    uint64_t total = 0;

    bool open(const std::string & path) {
        abort();
        path_ = path;
        used  = 0;
        total = 0;
        bad   = false;
#ifdef _WIN32
        const std::wstring w = fs::path(path).wstring();
        h = CreateFileW(w.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS,
                        FILE_ATTRIBUTE_NORMAL | FILE_FLAG_NO_BUFFERING | FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
        if (h == INVALID_HANDLE_VALUE) {
            h = CreateFileW(w.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
        }
        return h != INVALID_HANDLE_VALUE;
#else
        f.open(path, std::ios::binary | std::ios::trunc);
        return (bool) f;
#endif
    }

    bool write(const void * src, size_t n) {
        const uint8_t * in = (const uint8_t *) src;
        while (n > 0 && !bad) {
            const size_t take = std::min(n, spc_stage::SIZE - used);
            memcpy(st.p + used, in, take);
            used  += take;
            in    += take;
            n     -= take;
            total += take;
            if (used == spc_stage::SIZE) {
                flush(spc_stage::SIZE);
            }
        }
        return !bad;
    }

    // the tail goes out padded to the sector size and the file is then cut back to its real length
    bool close() {
        if (!is_open()) {
            return false;
        }
        if (used > 0 && !bad) {
            // 64 KiB covers every sector size in use, 4Kn included, and the stage is 8 MiB
            const size_t padded = (used + 65535) & ~(size_t) 65535;
            memset(st.p + used, 0, padded - used);
            flush(padded);
        }
#ifdef _WIN32
        if (!bad) {
            FILE_END_OF_FILE_INFO eof {};
            eof.EndOfFile.QuadPart = (LONGLONG) total;
            if (!SetFileInformationByHandle(h, FileEndOfFileInfo, &eof, sizeof(eof))) {
                bad = true;
            }
        }
        CloseHandle(h);
        h = INVALID_HANDLE_VALUE;
#else
        if (!bad) {
            f.flush();
            bad = !f;
        }
        f.close();
#endif
        return !bad;
    }

    // close and remove what was written
    void abort() {
        if (!is_open()) {
            return;
        }
#ifdef _WIN32
        CloseHandle(h);
        h = INVALID_HANDLE_VALUE;
#else
        f.close();
#endif
        std::error_code ec;
        fs::remove(path_, ec);
    }

private:
    spc_stage & st;
    size_t      used = 0;
    bool        bad  = false;
    std::string path_;
#ifdef _WIN32
    HANDLE h = INVALID_HANDLE_VALUE;
    bool is_open() const { return h != INVALID_HANDLE_VALUE; }
#else
    std::ofstream f;
    bool is_open() const { return f.is_open(); }
#endif

    void flush(size_t n) {
#ifdef _WIN32
        DWORD put = 0;
        if (!WriteFile(h, st.p, (DWORD) n, &put, nullptr) || put != n) {
            bad = true;
        }
#else
        f.write((const char *) st.p, (std::streamsize) std::min(n, used));
        bad = bad || !f;
#endif
        used = 0;
    }
};

template <typename T> bool put(spc_out & o, const T & v) { return o.write(&v, sizeof(v)); }
template <typename T> bool get(spc_file & f, T & v) { return f.read(&v, sizeof(v)); }

bool put_blob(spc_out & o, const std::vector<uint8_t> & v) {
    return put(o, (uint64_t) v.size()) && (v.empty() || o.write(v.data(), v.size()));
}

// a load splits "read" into the file and the buffer it goes into; per thread, the writer converts too
struct spc_read_cost { int64_t alloc_us; int64_t read_us; uint64_t bytes; };
thread_local spc_read_cost g_spc_cost = {};

bool get_blob(spc_file & f, std::vector<uint8_t> & v, uint64_t limit) {
    uint64_t n = 0;
    if (!get(f, n) || n > limit) {
        return false;
    }
    const int64_t t0 = ggml_time_us();
    v.resize(n);
    const int64_t t1 = ggml_time_us();
    const bool ok = n == 0 || f.read(v.data(), n);
    g_spc_cost.alloc_us += t1 - t0;
    g_spc_cost.read_us  += ggml_time_us() - t1;
    g_spc_cost.bytes    += n;
    return ok;
}

bool put_tokens(spc_out & o, const server_tokens & t) {
    const std::vector<char> b = t.serialize();
    return put(o, (uint64_t) b.size()) && (b.empty() || o.write(b.data(), b.size()));
}

// tokens as the serialized bytes, reinterpreted the way slot restore does
bool get_tokens(spc_file & f, server_tokens & out, bool has_mtmd) {
    std::vector<uint8_t> raw;
    if (!get_blob(f, raw, 1ull << 30) || raw.size() % sizeof(llama_token) != 0) {
        return false;
    }
    llama_tokens packed(raw.size() / sizeof(llama_token));
    if (!packed.empty()) {
        memcpy(packed.data(), raw.data(), raw.size());
    }
    out = server_tokens::deserialize(packed, has_mtmd);
    return true;
}

// write a file as <path>.part; nothing is left behind on failure
template <typename F> bool write_part(spc_stage & st, const std::string & part, F && body, uint64_t & bytes) {
    spc_out o(st);
    if (!o.open(part)) {
        return false;
    }
    if (!body(o) || !o.close()) {
        o.abort();
        std::error_code ec;
        fs::remove(part, ec);
        return false;
    }
    bytes = o.total;
    return true;
}

bool place(const std::string & part, const std::string & path) {
    std::error_code ec;
    fs::rename(part, path, ec);
    if (ec) {
        fs::remove(part, ec);
        return false;
    }
    return true;
}

std::string manifest_path(const std::string & dir, const server_tokens & tokens) {
    const std::vector<char> b = tokens.serialize();
    char name[64];
    snprintf(name, sizeof(name), "%016llx-%d.spc", (unsigned long long) XXH3_64bits(b.data(), b.size()), (int) tokens.size());
    return (fs::path(dir) / name).string();
}

bool write_manifest_part(spc_stage & st, const std::string & part, const disk_entry & e, bool has_mtmd, uint64_t & bytes) {
    const uint32_t version = e.version == (int32_t) SPC_V3 ? SPC_V3 : SPC_V2;
    return write_part(st, part, [&](spc_out & o) {
        bool ok = o.write(SPC_MAGIC, 8) && put(o, version) && put(o, (uint32_t) (has_mtmd ? 1 : 0))
               && put_tokens(o, e.tokens) && put(o, (uint32_t) e.ckpts.size());
        for (const auto & k : e.ckpts) {
            ok = ok && put(o, k.key) && put(o, k.n_tokens) && put(o, k.pos_min) && put(o, k.pos_max)
                    && put(o, k.size_tgt) && put(o, k.size_dft) && put(o, k.size_spec);
        }
        if (version == SPC_V3) {
            for (int b = 0; b < 2; ++b) {
                const auto & runs = b == 0 ? e.runs_tgt : e.runs_dft;
                ok = ok && put(o, b == 0 ? e.row_tgt : e.row_dft) && put(o, (uint32_t) runs.size());
                for (const auto & r : runs) {
                    ok = ok && put(o, r.hi) && put(o, r.lo) && put(o, r.pos0) && put(o, r.n) && put(o, r.n_use);
                }
            }
            return ok;
        }
        for (int b = 0; b < 2; ++b) {
            const auto & refs = b == 0 ? e.main : e.drft;
            ok = ok && put(o, b == 0 ? e.main_size : e.drft_size) && put(o, (uint32_t) refs.size());
            for (const auto & c : refs) {
                ok = ok && put(o, c.hi) && put(o, c.lo) && put(o, c.size);
            }
        }
        return ok;
    }, bytes);
}

// a run file: the rows as the context gave them, checked against their name on the way back
bool write_run_body(spc_out & o, uint64_t row, const disk_run_ref & r, const std::vector<uint8_t> & rows) {
    return o.write(RUN_MAGIC, 8) && put(o, RUN_V1) && put(o, row) && put(o, (uint32_t) r.n)
        && o.write(rows.data(), rows.size()) && put(o, r.hi) && put(o, r.lo);
}

// into buf, resized to the run's rows; false for a missing, damaged or foreign file. Throws std::bad_alloc when
// there is no memory for it, which says nothing about the file.
bool read_run(spc_file & f, const std::string & path, const disk_run_ref & r, uint64_t row, std::vector<uint8_t> & buf) {
    if (!f.open(path)) {
        return false;
    }
    char magic[8];
    uint32_t version = 0, n = 0;
    uint64_t row_ref = 0, hi = 0, lo = 0;
    const size_t size = (size_t) row * (size_t) r.n;
    bool ok = f.read(magic, 8) && memcmp(magic, RUN_MAGIC, 8) == 0 && get(f, version) && version == RUN_V1
           && get(f, row_ref) && row_ref == row && get(f, n) && (int32_t) n == r.n;
    if (ok) {
        const int64_t t0 = ggml_time_us();
        buf.resize(size);
        const int64_t t1 = ggml_time_us();
        ok = f.read(buf.data(), size) && get(f, hi) && get(f, lo);
        g_spc_cost.alloc_us += t1 - t0;
        g_spc_cost.read_us  += ggml_time_us() - t1;
        g_spc_cost.bytes    += size;
    }
    f.close();
    if (ok) {
        const XXH128_hash_t h = XXH3_128bits(buf.data(), size);
        ok = h.high64 == r.hi && h.low64 == r.lo && hi == r.hi && lo == r.lo;
    }
    return ok;
}

// the header of an entry: its tokens and the objects it names. `foreign` says the file is not to be deleted
// for failing - it could not be opened (a lock, a scanner) or a later build wrote it; `old` that it is of a
// format before `version`, which this store no longer holds
bool scan_entry(spc_stage & st, disk_entry & e, bool has_mtmd, int32_t version_cur, std::string & why, bool & foreign,
                bool & old) {
    foreign = false;
    old     = false;
    spc_file f(st);
    if (!f.open(e.path)) {
        why = "cannot open it";
        foreign = true;
        return false;
    }
    char magic[8];
    uint32_t version = 0, mtmd = 0;
    if (!f.read(magic, 8) || memcmp(magic, SPC_MAGIC, 8) != 0 || !get(f, version) || !get(f, mtmd)) {
        why = "not a cache file";
        return false;
    }
    if (version > SPC_V3) {
        why = "format version " + std::to_string(version);
        foreign = true;
        return false;
    }
    if ((int32_t) version < version_cur) {
        why = "format version " + std::to_string(version);
        old = true;
        return false;
    }
    e.version = (int32_t) version;
    try {
        if (!get_tokens(f, e.tokens, has_mtmd)) {
            why = "truncated or malformed token state";
            return false;
        }
    } catch (const std::exception & ex) {
        why = ex.what();
        return false;
    }
    // a manifest is named after its tokens: a mismatch is a damaged file, found for free
    {
        const std::vector<char> b = e.tokens.serialize();
        char name[32];
        snprintf(name, sizeof(name), "%016llx-", (unsigned long long) XXH3_64bits(b.data(), b.size()));
        if (fs::path(e.path).filename().string().rfind(name, 0) != 0) {
            why = "its tokens do not match its name";
            return false;
        }
    }
    uint32_t n_ckpt = 0;
    if (!get(f, n_ckpt) || n_ckpt > 4096) {
        why = "bad checkpoint list";
        return false;
    }
    for (uint32_t i = 0; i < n_ckpt; ++i) {
        disk_ckpt_ref r {};
        if (!get(f, r.key) || !get(f, r.n_tokens) || !get(f, r.pos_min) || !get(f, r.pos_max) ||
            !get(f, r.size_tgt) || !get(f, r.size_dft) || !get(f, r.size_spec)) {
            why = "truncated checkpoint list";
            return false;
        }
        e.ckpts.push_back(r);
    }
    if (version == SPC_V3) {
        // runs from position 0, one after another: a gap or an overlap is a damaged manifest
        for (int b = 0; b < 2; ++b) {
            auto & runs = b == 0 ? e.runs_tgt : e.runs_dft;
            uint64_t & row = b == 0 ? e.row_tgt : e.row_dft;
            uint32_t n = 0;
            if (!get(f, row) || !get(f, n) || n > (1u << 20) || row > (1u << 24)) {
                why = "bad run list";
                return false;
            }
            int64_t next = 0;
            for (uint32_t i = 0; i < n; ++i) {
                disk_run_ref r {};
                if (!get(f, r.hi) || !get(f, r.lo) || !get(f, r.pos0) || !get(f, r.n) || !get(f, r.n_use)) {
                    why = "truncated run list";
                    return false;
                }
                if (r.pos0 != next || r.n <= 0 || r.n_use <= 0 || r.n_use > r.n || row == 0) {
                    why = "runs out of order";
                    return false;
                }
                next += r.n_use;
                runs.push_back(r);
            }
        }
        e.n_exact = runs_exact(e.tokens, e.runs_tgt, e.runs_dft, e.row_dft > 0, e.ckpts);
        return true;
    }
    for (int b = 0; b < 2; ++b) {
        auto & refs = b == 0 ? e.main : e.drft;
        uint64_t & size = b == 0 ? e.main_size : e.drft_size;
        uint32_t n = 0;
        if (!get(f, size) || !get(f, n) || n > (1u << 24)) {
            why = "bad chunk list";
            return false;
        }
        uint64_t sum = 0;
        for (uint32_t i = 0; i < n; ++i) {
            disk_chunk_ref c {};
            if (!get(f, c.hi) || !get(f, c.lo) || !get(f, c.size)) {
                why = "truncated chunk list";
                return false;
            }
            if (c.size == 0 || c.size > CDC_MAX) {
                why = "chunk size out of range";
                return false;
            }
            refs.push_back(c);
            sum += c.size;
        }
        if (sum != size) {
            why = "chunk sizes do not add up";
            return false;
        }
    }
    return true;
}

// checkpoint file, version 2: the version 1 layout followed by an XXH3-128 of the three payloads. A
// checkpoint is more than half of an entry's bytes, and one wrong byte in a recurrent state restores
// nonsense or trips the size assert in the restore, so it is checked like a chunk.
XXH128_hash_t ckpt_hash(const common_prompt_checkpoint & c) {
    XXH3_state_t hs;
    XXH3_128bits_reset(&hs);
    for (const auto * v : { &c.data_tgt, &c.data_dft, &c.data_spec }) {
        if (!v->empty()) {
            XXH3_128bits_update(&hs, v->data(), v->size());
        }
    }
    return XXH3_128bits_digest(&hs);
}

bool write_ckpt_body(spc_out & o, const common_prompt_checkpoint & c) {
    const XXH128_hash_t h = ckpt_hash(c);
    return o.write(CKP_MAGIC, 8) && put(o, CKP_V2) && put(o, (int64_t) c.n_tokens) && put(o, (int32_t) c.pos_min)
        && put(o, (int32_t) c.pos_max) && put_blob(o, c.data_tgt) && put_blob(o, c.data_dft) && put_blob(o, c.data_spec)
        && put(o, h.high64) && put(o, h.low64);
}

// every blob is read against the size the manifest recorded, so a damaged length cannot ask for a TiB
bool read_ckpt(spc_file & f, const std::string & path, const disk_ckpt_ref & r, common_prompt_checkpoint & c) {
    if (!f.open(path)) {
        return false;
    }
    char magic[8];
    uint32_t version = 0;
    int64_t n_tokens = 0;
    int32_t pmin = 0, pmax = 0;
    uint64_t hi = 0, lo = 0;
    bool ok = f.read(magic, 8) && memcmp(magic, CKP_MAGIC, 8) == 0 && get(f, version) && version == CKP_V2
        && get(f, n_tokens) && get(f, pmin) && get(f, pmax)
        && n_tokens == r.n_tokens && pmin == r.pos_min && pmax == r.pos_max
        && get_blob(f, c.data_tgt,  r.size_tgt)  && c.data_tgt.size()  == r.size_tgt
        && get_blob(f, c.data_dft,  r.size_dft)  && c.data_dft.size()  == r.size_dft
        && get_blob(f, c.data_spec, r.size_spec) && c.data_spec.size() == r.size_spec
        && get(f, hi) && get(f, lo);
    f.close();
    if (ok) {
        const XXH128_hash_t h = ckpt_hash(c);
        ok = h.high64 == hi && h.low64 == lo;
    }
    if (ok) {
        c.n_tokens = n_tokens;
        c.pos_min  = pmin;
        c.pos_max  = pmax;
    }
    return ok;
}

// what a failed read blames, so the writer can retire it along with every entry that names it
struct read_fault {
    bool     chunk = false;
    uint64_t hi = 0, lo = 0;
    bool     ckpt = false;
    uint64_t key = 0;
    bool     run = false;              // hi, lo name it
};

// chunks are separate files, and one reader leaves the drive idle between them: 2.8 GB/s from one thread
// against 6.5 GB/s from four, measured unbuffered on this machine (eight add nothing)
constexpr int SPC_READERS = 4;

// an entry into memory. Chunks and checkpoints are checked against their hashes: a state that is wrong in
// one byte decodes into nonsense, and XXH3 over 5.6 GiB costs ~0.2 s. With `lazy_ckpts` a version 2 entry's
// checkpoints are left on disk and come back as their headers only (no bytes); the caller pins and pages them.
// Throws std::bad_alloc when there is no memory for it, which says nothing about the entry.
bool read_entry(spc_stage & st, const std::string & dir, const disk_entry & e,
                server_prompt_cache_state & state, std::string & why, read_fault & fault, bool lazy_ckpts = false) {
    spc_file f(st);
    state.prompt.tokens = e.tokens.clone();
    auto fill = [&](std::vector<uint8_t> & v, uint64_t size, const std::vector<disk_chunk_ref> & refs) -> bool {
        const int64_t a0 = ggml_time_us();
        v.resize(size);
        g_spc_cost.alloc_us += ggml_time_us() - a0;
        std::vector<uint64_t> offs(refs.size());
        uint64_t off = 0;
        for (size_t i = 0; i < refs.size(); ++i) {
            if (off + refs[i].size > size) {
                why = "chunk list longer than the state";
                return false;
            }
            offs[i] = off;
            off += refs[i].size;
        }
        if (off != size) {
            why = "chunk list shorter than the state";
            return false;
        }
        // the chunks land in disjoint ranges of v; each reader has its own staging buffer, and this thread
        // uses the caller's. A reader that cannot get a buffer leaves its share to the others.
        std::atomic<size_t>   next   { 0 };
        std::atomic<bool>     failed { false };
        std::atomic<uint64_t> bytes  { 0 };
        std::mutex            fault_mu;
        auto reader = [&](spc_file * fp) {
            std::unique_ptr<spc_stage> own_st;
            std::unique_ptr<spc_file>  own_f;
            if (!fp) {
                try {
                    own_st = std::make_unique<spc_stage>();
                } catch (const std::bad_alloc &) {
                    return;
                }
                own_f = std::make_unique<spc_file>(*own_st);
                fp = own_f.get();
            }
            for (size_t i; !failed && (i = next.fetch_add(1)) < refs.size(); ) {
                const auto & c = refs[i];
                const bool ok = fp->open(chunk_path(dir, c.hi, c.lo)) && fp->read(v.data() + offs[i], c.size);
                fp->close();
                const disk_chunk_ref got = ok ? chunk_ref(v.data() + offs[i], c.size) : disk_chunk_ref {};
                if (!ok || got.hi != c.hi || got.lo != c.lo) {
                    std::lock_guard<std::mutex> lk(fault_mu);
                    if (!failed.exchange(true)) {
                        why = (ok ? "chunk does not match its name: " : "missing chunk ") + chunk_hex(c.hi, c.lo);
                        fault.chunk = true;
                        fault.hi = c.hi;
                        fault.lo = c.lo;
                    }
                    return;
                }
                bytes += c.size;
            }
        };
        const int64_t r0 = ggml_time_us();
        std::vector<std::thread> pool;
        for (int t = 1; t < SPC_READERS && (size_t) t < refs.size(); ++t) {
            try {
                pool.emplace_back(reader, nullptr);
            } catch (const std::system_error &) {
                break;                                  // fewer readers, same result
            }
        }
        reader(&f);
        for (auto & t : pool) {
            t.join();
        }
        g_spc_cost.read_us += ggml_time_us() - r0;
        g_spc_cost.bytes   += bytes;
        return !failed;
    };
    if (!fill(state.data.main, e.main_size, e.main) || !fill(state.data.drft, e.drft_size, e.drft)) {
        return false;
    }
    for (const auto & k : e.ckpts) {
        common_prompt_checkpoint c;
        if (lazy_ckpts) {
            c.n_tokens = k.n_tokens;                    // the header: a rewind reads the bytes if it picks this one
            c.pos_min  = k.pos_min;
            c.pos_max  = k.pos_max;
        } else if (!read_ckpt(f, ckpt_path(dir, k.key), k, c)) {       // its blobs time themselves
            why = "missing or damaged checkpoint";
            fault.ckpt = true;
            fault.key  = k.key;
            return false;
        }
        state.prompt.checkpoints.push_back(std::move(c));
    }
    return true;
}

int64_t mtime_ms(const fs::directory_entry & e) {
    std::error_code ec;
    return std::chrono::duration_cast<std::chrono::milliseconds>(e.last_write_time(ec).time_since_epoch()).count();
}

// the files in a directory, without the throwing overloads: one bad directory entry must not take the
// server down at startup
std::vector<fs::directory_entry> list_dir(const fs::path & dir, bool recursive) {
    std::vector<fs::directory_entry> out;
    std::error_code ec;
    if (recursive) {
        for (auto it = fs::recursive_directory_iterator(dir, ec); !ec && it != fs::recursive_directory_iterator(); it.increment(ec)) {
            std::error_code ec2;
            if (it->is_regular_file(ec2)) {
                out.push_back(*it);
            }
        }
    } else {
        for (auto it = fs::directory_iterator(dir, ec); !ec && it != fs::directory_iterator(); it.increment(ec)) {
            std::error_code ec2;
            if (it->is_regular_file(ec2)) {
                out.push_back(*it);
            }
        }
    }
    return out;
}

} // namespace

server_prompt_cache::~server_prompt_cache() {
    if (disk_thread.joinable()) {
        {
            std::lock_guard<std::mutex> lk(disk_mu);
            disk_stop = true;
            disk_on_written = nullptr;     // nothing to wake once the server is going away
        }
        disk_cv.notify_all();
        disk_thread.join();
    }
}

void server_prompt_cache::set_disk(const std::string & root, size_t limit_mib, bool has_mtmd, const std::string & model_tag,
                                   int32_t version, uint64_t row_tgt, uint64_t row_dft) {
    std::error_code ec;
    // one directory per model: a state is only meaningful to the model that computed it, and two quants of
    // one model have identical state shapes, so the restore would take the other's without a word
    const fs::path dir = model_tag.empty() ? fs::path(root) : fs::path(root) / model_tag;
    if (!model_tag.empty() && !fs::exists(dir, ec)) {
        // entries written before caches were kept per model sit in the root: they are this model's as far as
        // anything can tell, and the first model to start takes them
        std::vector<fs::path> legacy;
        for (const auto & de : list_dir(root, false)) {
            if (de.path().extension() == ".spc") {
                legacy.push_back(de.path());
            }
        }
        fs::create_directories(dir, ec);
        int moved = 0;
        for (const auto & p : legacy) {
            fs::rename(p, dir / p.filename(), ec);
            moved += !ec;
        }
        for (const char * sub : { "chunks", "ckpt", "runs" }) {
            if (fs::exists(fs::path(root) / sub, ec)) {
                fs::rename(fs::path(root) / sub, dir / sub, ec);
            }
        }
        if (moved > 0) {
            SRV_INF(" - disk cache: %d entries written before caches were kept per model moved into %s\n",
                    moved, dir.string().c_str());
        }
    }
    fs::create_directories(dir / "ckpt", ec);
    fs::create_directories(dir / "chunks", ec);
    fs::create_directories(dir / "runs", ec);
    disk_dir      = dir.string();
    disk_limit    = (uint64_t) limit_mib * 1024ull * 1024ull;
    disk_has_mtmd = has_mtmd;
    disk_version  = version;
    if (const char * b = getenv("STRIX_PROMPT_CACHE_BLOCK"); b && *b) {
        disk_block = std::max(0, atoi(b));
    }
    if (const char * r = getenv("STRIX_PROMPT_CACHE_RUN"); r && *r) {
        disk_run = std::max(0, atoi(r));
    }

    std::lock_guard<std::mutex> lk(disk_mu);
    disk_index.clear();
    disk_chunks.clear();
    disk_ckpts.clear();
    disk_runs.clear();
    disk_bytes = 0;

    // 1. what an interrupted write left
    for (const auto & de : list_dir(dir, true)) {
        if (de.path().extension() == ".part") {
            fs::remove(de.path(), ec);
        }
    }

    // 2. entries, and a reference for each object they name. One of an earlier format goes, and the objects
    // only it named are swept below as orphans. A file this build cannot open or does not know the version
    // of is left alone - a later build may have written it - and then nothing is swept, because its objects
    // cannot be told apart from orphans.
    spc_stage stage;
    int n_skipped = 0;
    int n_old = 0;
    int n_no_draft = 0;
    uint64_t old_bytes = 0;
    for (const auto & de : list_dir(dir, false)) {
        if (de.path().extension() != ".spc") {
            continue;
        }
        disk_entry e;
        e.path  = de.path().string();
        e.bytes = (uint64_t) de.file_size(ec);
        e.order = mtime_ms(de);
        std::string why;
        bool foreign = false, old = false;
        bool ok = scan_entry(stage, e, disk_has_mtmd, disk_version, why, foreign, old);
        // strixllama: a conversation saved with MTP off has no draft rows. It went with the rows of another size
        // below, so turning MTP back on deleted every conversation saved meanwhile (7 of them, 3.6 GiB, 10-09). It
        // stays: the lookup passes it over while this server drafts (load_runs), and it is there again with MTP off
        if (ok && e.version == (int32_t) SPC_V3 && row_tgt > 0 && e.row_tgt == row_tgt && row_dft > 0 && e.row_dft == 0) {
            ++n_no_draft;
        } else if (ok && e.version == (int32_t) SPC_V3 && row_tgt > 0 && (e.row_tgt != row_tgt || (row_dft > 0 && e.row_dft != row_dft))) {
            // rows of another size - a K/V cache of another type, another draft - are as unusable as an older
            // format, and go the same way
            ok = false;
            old = true;
        }
        if (!ok) {
            if (old) {
                fs::remove(de.path(), ec);
                ++n_old;
                old_bytes += e.bytes;
            } else if (foreign) {
                SRV_WRN(" - disk cache: leaving %s alone: %s\n", de.path().filename().string().c_str(), why.c_str());
                ++n_skipped;
            } else {
                SRV_WRN(" - disk cache: dropping %s (%.2f GiB): %s\n", de.path().filename().string().c_str(),
                        e.bytes / (1024.0 * 1024.0 * 1024.0), why.c_str());
                fs::remove(de.path(), ec);
            }
            continue;
        }
        disk_seq = std::max(disk_seq, e.order);
        for (const auto * refs : { &e.main, &e.drft }) {
            for (const auto & c : *refs) {
                disk_chunks[{ c.hi, c.lo }].refs++;
            }
        }
        for (const auto * runs : { &e.runs_tgt, &e.runs_dft }) {
            for (const auto & r : *runs) {
                disk_runs[{ r.hi, r.lo }].refs++;
            }
        }
        for (const auto & k : e.ckpts) {
            disk_ckpts[k.key].refs++;
        }
        disk_index.push_back(std::move(e));
    }

    // 3. objects: size what is referenced, remove what is not (unless a skipped entry might name it)
    size_t n_orphans = 0;
    uint64_t orphan_bytes = 0;
    for (const auto & de : list_dir(dir / "chunks", true)) {
        if (de.path().extension() != ".chk") {
            continue;
        }
        const std::string hex = de.path().stem().string();
        uint64_t hi = 0, lo = 0;
        bool named = hex.size() == 32;
        if (named) {
            char * end = nullptr;
            hi = strtoull(hex.substr(0, 16).c_str(), &end, 16); named = named && *end == 0;
            lo = strtoull(hex.substr(16).c_str(), &end, 16);    named = named && *end == 0;
        }
        auto o = named ? disk_chunks.find({ hi, lo }) : disk_chunks.end();
        if (o != disk_chunks.end()) {
            o->second.bytes = (uint64_t) de.file_size(ec);
        } else if (n_skipped == 0) {
            orphan_bytes += (uint64_t) de.file_size(ec);
            fs::remove(de.path(), ec);
            ++n_orphans;
        }
    }
    for (const auto & de : list_dir(dir / "ckpt", false)) {
        if (de.path().extension() != ".ckp") {
            continue;
        }
        char * end = nullptr;
        const uint64_t key = strtoull(de.path().stem().string().c_str(), &end, 16);
        auto o = *end == 0 ? disk_ckpts.find(key) : disk_ckpts.end();
        if (o != disk_ckpts.end()) {
            o->second.bytes = (uint64_t) de.file_size(ec);
        } else if (n_skipped == 0) {
            orphan_bytes += (uint64_t) de.file_size(ec);
            fs::remove(de.path(), ec);
            ++n_orphans;
        }
    }
    for (const auto & de : list_dir(dir / "runs", true)) {
        if (de.path().extension() != ".run") {
            continue;
        }
        const std::string hex = de.path().stem().string();
        uint64_t hi = 0, lo = 0;
        bool named = hex.size() == 32;
        if (named) {
            char * end = nullptr;
            hi = strtoull(hex.substr(0, 16).c_str(), &end, 16); named = named && *end == 0;
            lo = strtoull(hex.substr(16).c_str(), &end, 16);    named = named && *end == 0;
        }
        auto o = named ? disk_runs.find({ hi, lo }) : disk_runs.end();
        if (o != disk_runs.end()) {
            o->second.bytes = (uint64_t) de.file_size(ec);
        } else if (n_skipped == 0) {
            orphan_bytes += (uint64_t) de.file_size(ec);
            fs::remove(de.path(), ec);
            ++n_orphans;
        }
    }

    // 4. an entry whose objects are gone, or not the size it recorded, cannot be restored
    for (auto it = disk_index.begin(); it != disk_index.end();) {
        bool whole = true;
        for (const auto * refs : { &it->main, &it->drft }) {
            for (const auto & c : *refs) {
                whole = whole && disk_chunks[{ c.hi, c.lo }].bytes == c.size;
            }
        }
        for (int b = 0; b < 2; ++b) {
            for (const auto & r : b == 0 ? it->runs_tgt : it->runs_dft) {
                whole = whole && disk_runs[{ r.hi, r.lo }].bytes == run_file_bytes(b == 0 ? it->row_tgt : it->row_dft, r.n);
            }
        }
        for (const auto & k : it->ckpts) {
            whole = whole && disk_ckpts[k.key].bytes > 0;
        }
        if (whole) {
            ++it;
            continue;
        }
        SRV_WRN(" - disk cache: dropping %s: objects it names are missing or damaged\n", fs::path(it->path).filename().string().c_str());
        disk_drop(*it);
        it = disk_index.erase(it);
    }
    disk_bytes = 0;
    for (const auto & e : disk_index) {
        disk_bytes += e.bytes;
    }
    for (const auto & [key, o] : disk_chunks) {
        disk_bytes += o.bytes;
    }
    for (const auto & [key, o] : disk_ckpts) {
        disk_bytes += o.bytes;
    }
    for (const auto & [key, o] : disk_runs) {
        disk_bytes += o.bytes;
    }
    disk_evict("");

    // 5. the writer
    disk_stop = false;
    disk_thread = std::thread([this] { disk_writer(); });

    if (n_old > 0) {
        SRV_INF(" - disk cache: %d entries of a format before version %d or of another K/V cache type removed, %.2f GiB with the objects no entry names any more\n",
                n_old, disk_version, (old_bytes + orphan_bytes) / (1024.0 * 1024.0 * 1024.0));
    }
    if (n_no_draft > 0) {
        SRV_INF(" - disk cache: %d conversations saved with MTP off are kept, and used again with MTP off\n", n_no_draft);
    }
    char orphans[64] = "";
    if (n_orphans > 0) {
        snprintf(orphans, sizeof(orphans), ", %zu orphans removed (%.2f GiB)", n_orphans, orphan_bytes / (1024.0 * 1024.0 * 1024.0));
    }
    SRV_INF("disk prompt cache: %zu entries (version %d), %.1f GiB in %s (limit %.1f GiB, %zu chunks, %zu runs, %zu checkpoints, "
            "write every %d tokens of growth, runs of %d%s%s)\n",
            disk_index.size(), disk_version, disk_bytes / (1024.0 * 1024.0 * 1024.0), disk_dir.c_str(), disk_limit / (1024.0 * 1024.0 * 1024.0),
            disk_chunks.size(), disk_runs.size(), disk_ckpts.size(), disk_block, disk_run, orphans,
            n_skipped ? (", " + std::to_string(n_skipped) + " entries left alone").c_str() : "");
}

uint64_t server_prompt_cache::disk_size() {
    std::lock_guard<std::mutex> lk(disk_mu);
    return disk_bytes;
}

bool server_prompt_cache::disk_busy() {
    std::lock_guard<std::mutex> lk(disk_mu);
    return !disk_queue.empty();
}

bool server_prompt_cache::disk_drain(int64_t timeout_ms) {
    std::unique_lock<std::mutex> lk(disk_mu);
    return disk_cv.wait_for(lk, std::chrono::milliseconds(timeout_ms), [&] { return disk_queue.empty() && disk_current == nullptr; });
}

// caller holds disk_mu. The entry's objects lose a reference each; those nobody holds any more go, and so
// does the entry's own file unless `remove_file` is false (its path now holds a newer manifest). Only the
// writer thread calls this (and set_disk, before the writer starts).
void server_prompt_cache::disk_drop(const disk_entry & e, bool remove_file) {
    std::error_code ec;
    for (const auto * refs : { &e.main, &e.drft }) {
        for (const auto & c : *refs) {
            auto it = disk_chunks.find({ c.hi, c.lo });
            if (it == disk_chunks.end()) {
                continue;
            }
            if (--it->second.refs <= 0) {
                fs::remove(chunk_path(disk_dir, c.hi, c.lo), ec);
                disk_bytes -= std::min(disk_bytes, it->second.bytes);
                disk_chunks.erase(it);
            }
        }
    }
    for (const auto & k : e.ckpts) {
        auto it = disk_ckpts.find(k.key);
        if (it == disk_ckpts.end()) {
            continue;
        }
        if (--it->second.refs <= 0) {
            fs::remove(ckpt_path(disk_dir, k.key), ec);
            disk_bytes -= std::min(disk_bytes, it->second.bytes);
            disk_ckpts.erase(it);
        }
    }
    for (const auto * runs : { &e.runs_tgt, &e.runs_dft }) {
        for (const auto & r : *runs) {
            auto it = disk_runs.find({ r.hi, r.lo });
            if (it == disk_runs.end()) {
                continue;
            }
            if (--it->second.refs <= 0) {
                fs::remove(run_path(disk_dir, r.hi, r.lo), ec);
                disk_bytes -= std::min(disk_bytes, it->second.bytes);
                disk_runs.erase(it);
            }
        }
    }
    if (remove_file) {
        fs::remove(e.path, ec);
    }
    disk_bytes -= std::min(disk_bytes, e.bytes);
}

// caller holds disk_mu: least recently written or used goes first, never `keep`
void server_prompt_cache::disk_evict(const std::string & keep) {
    while (disk_index.size() > 1 && disk_bytes > disk_limit) {
        auto oldest = disk_index.end();
        for (auto it = disk_index.begin(); it != disk_index.end(); ++it) {
            if (it->path != keep && (oldest == disk_index.end() || it->order < oldest->order)) {
                oldest = it;
            }
        }
        if (oldest == disk_index.end()) {
            break;
        }
        SRV_INF(" - disk cache: evicting %d tokens (%s) to stay under %.1f GiB\n", (int) oldest->tokens.size(),
                fs::path(oldest->path).filename().string().c_str(), disk_limit / (1024.0 * 1024.0 * 1024.0));
        disk_drop(*oldest);
        disk_index.erase(oldest);
    }
}

// caller holds disk_mu
size_t server_prompt_cache::disk_covered_locked(const server_tokens & tokens, bool with_jobs) {
    size_t best = 0;
    // exact < 0: all of t comes back (versions 1 and 2); else its first `exact` tokens do (version 3)
    auto consider = [&](const server_tokens & t, int64_t exact) {
        const size_t lcp = t.get_common_prefix(tokens);
        if (exact >= 0) {
            const size_t e = std::min((size_t) exact, tokens.size());
            if (lcp >= e) {
                best = std::max(best, e);
            }
            return;
        }
        if (lcp == tokens.size()) {
            best = tokens.size();                   // something stored starts with all of it
        } else if (lcp == t.size()) {
            best = std::max(best, lcp);             // all of something stored is a prefix of it
        }
    };
    for (const auto & e : disk_index) {
        consider(e.tokens, e.version == (int32_t) SPC_V3 ? e.n_exact : -1);
        if (best == tokens.size()) {
            return best;
        }
    }
    if (with_jobs) {
        for (const auto & j : disk_queue) {
            consider(j->tokens, j->version == (int32_t) SPC_V3 ? j->n_exact : -1);
        }
        if (disk_current) {
            consider(disk_current->tokens, disk_current->version == (int32_t) SPC_V3 ? disk_current->n_exact : -1);
        }
    }
    return best;
}

size_t server_prompt_cache::disk_exact(const server_tokens & tokens) {
    return disk_covered(tokens, /* with_jobs = */ true);
}

server_prompt_cache::disk_run_ref server_prompt_cache::make_run_ref(const std::vector<uint8_t> & rows, int32_t pos0, int32_t n) {
    const XXH128_hash_t h = XXH3_128bits(rows.data(), rows.size());
    return { h.high64, h.low64, pos0, n, n };
}

bool server_prompt_cache::runs_lost(int32_t id_slot) {
    std::lock_guard<std::mutex> lk(disk_mu);
    return disk_runs_lost.erase(id_slot) > 0;
}

size_t server_prompt_cache::disk_covered(const server_tokens & tokens, bool with_jobs) {
    if (disk_limit == 0) {
        return 0;
    }
    std::lock_guard<std::mutex> lk(disk_mu);
    return disk_covered_locked(tokens, with_jobs);
}

// caller holds disk_mu: a job that is queued or being written and would be a better start for this prompt
// than what the slot or the RAM tier already offers (f_sim_base). Anything less is not worth waiting for -
// notably the slot's own previous conversation, queued a moment ago on its way out.
bool server_prompt_cache::disk_in_flight(const server_tokens & tokens, float f_sim_base) {
    if (!disk_thread.joinable() || disk_stop || tokens.size() == 0) {
        return false;
    }
    auto relevant = [&](const disk_job & j) {
        const size_t lcp = j.tokens.get_common_prefix(tokens);
        return lcp > 0 && lcp * 4 >= j.tokens.size() && float(lcp) / tokens.size() > f_sim_base;
    };
    for (const auto & j : disk_queue) {
        if (relevant(*j)) {
            return true;
        }
    }
    return disk_current && relevant(*disk_current);
}

// caller holds disk_mu
bool server_prompt_cache::disk_ckpt_in_flight(uint64_t key) {
    auto carries = [&](const disk_job & j) {
        return std::any_of(j.ckpt_data.begin(), j.ckpt_data.end(), [&](const auto & kc) { return kc.first == key; });
    };
    for (const auto & j : disk_queue) {
        if (carries(*j)) {
            return true;
        }
    }
    return disk_current && carries(*disk_current);
}

bool server_prompt_cache::persist(const server_prompt & prompt, std::vector<uint8_t> && data_main, std::vector<uint8_t> && data_drft, bool wait,
                                  const ckpt_paged_map * paged) {
    if (disk_limit == 0 || prompt.tokens.size() == 0 || !disk_thread.joinable()) {
        return false;
    }
    // media in the prompt is not persisted: the chunks would have to come back through an mmproj this
    // server may not have. get_text_tokens() drops the placeholders, so a short result is the tell.
    const llama_tokens text = prompt.tokens.get_text_tokens();
    if (text.size() != prompt.tokens.size()) {
        return false;
    }
    if (!wait && disk_busy()) {
        return false;                              // before copying GBs of checkpoints for nothing
    }

    auto job = std::make_unique<disk_job>();
    job->tokens = prompt.tokens.clone();
    job->main   = std::move(data_main);
    job->drft   = std::move(data_drft);

    // every checkpoint's ref, with the checkpoint itself when its bytes are here; one the slot handed back is
    // in the store under the ref it was paged out with - if that ref still belongs to these tokens
    std::vector<std::pair<disk_ckpt_ref, const common_prompt_checkpoint *>> refs;
    for (const auto & c : prompt.checkpoints) {
        if (c.data_tgt.empty()) {
            const auto p = paged ? paged->find({ c.n_tokens, c.pos_min, c.pos_max }) : ckpt_paged_map::const_iterator();
            if (paged && p != paged->end() &&
                ckpt_key(text, p->second.n_tokens, p->second.pos_min, p->second.pos_max,
                         p->second.size_tgt, p->second.size_dft, p->second.size_spec) == p->second.key) {
                refs.push_back({ p->second, nullptr });
            }
            continue;
        }
        refs.push_back({ { ckpt_key(text, c.n_tokens, c.pos_min, c.pos_max, c.data_tgt.size(), c.data_dft.size(), c.data_spec.size()),
                           c.n_tokens, c.pos_min, c.pos_max, c.data_tgt.size(), c.data_dft.size(), c.data_spec.size() }, &c });
    }

    // copy only the checkpoints the store has not got: they never change once made, so a conversation
    // that was saved before carries all of them but its newest. Whatever is named but not carried is
    // checked again by the writer just before the manifest is written, when it can no longer change.
    std::vector<bool> need(refs.size(), false);
    auto unpin_job = [&]() {
        std::lock_guard<std::mutex> lk(disk_mu);
        disk_unpin.insert(disk_unpin.end(), job->pinned.begin(), job->pinned.end());
        job->pinned.clear();
        disk_cv.notify_all();
    };
    {
        std::lock_guard<std::mutex> lk(disk_mu);
        for (size_t i = 0; i < refs.size(); ++i) {
            auto o = disk_ckpts.find(refs[i].first.key);
            const bool stored = o != disk_ckpts.end() || disk_ckpt_in_flight(refs[i].first.key);
            if (refs[i].second) {
                need[i] = !stored;
                job->ckpts.push_back(refs[i].first);
            } else if (stored) {
                job->ckpts.push_back(refs[i].first);
            }
            // named but not carried, and on disk: held until the job's entry counts it, so an eviction in the
            // meantime cannot take it
            if (!need[i] && o != disk_ckpts.end() && (refs[i].second || stored)) {
                o->second.refs++;
                job->pinned.push_back(refs[i].first.key);
            }
        }
    }
    try {
        for (size_t i = 0; i < refs.size(); ++i) {
            if (need[i]) {
                job->ckpt_data.emplace_back(refs[i].first.key, *refs[i].second);
            }
        }
    } catch (const std::bad_alloc &) {
        SRV_WRN("%s", " - disk cache: not enough memory to copy the checkpoints, not saving\n");
        unpin_job();
        return false;
    }

    std::unique_lock<std::mutex> lk(disk_mu);
    // one job waiting besides the one being written: each is a whole state in memory
    if (!disk_queue.empty()) {
        if (!wait || disk_stop) {
            lk.unlock();
            unpin_job();
            return false;
        }
        disk_cv.wait(lk, [&] { return disk_queue.empty() || disk_stop; });
        if (disk_stop) {
            lk.unlock();
            unpin_job();
            return false;
        }
    }
    disk_queue.push_back(std::move(job));
    disk_cv.notify_all();
    return true;
}

bool server_prompt_cache::persist_runs(int32_t id_slot, const server_tokens & tokens, const std::list<common_prompt_checkpoint> & ckpts,
                                       const std::set<const common_prompt_checkpoint *> & writable,
                                       const ckpt_paged_map * paged, const common_prompt_checkpoint * end, uint64_t row_tgt, uint64_t row_dft,
                                       const std::vector<disk_run_ref> & runs_tgt, const std::vector<disk_run_ref> & runs_dft,
                                       std::vector<std::pair<std::pair<uint64_t, uint64_t>, std::vector<uint8_t>>> && run_data, bool wait) {
    if (disk_limit == 0 || tokens.size() == 0 || !disk_thread.joinable() || row_tgt == 0) {
        return false;
    }
    const llama_tokens text = tokens.get_text_tokens();
    if (text.size() != tokens.size()) {
        return false;                              // media: never stored
    }
    if (!wait && disk_busy()) {
        return false;                              // the next round tries again
    }

    auto job = std::make_unique<disk_job>();
    job->version  = (int32_t) SPC_V3;
    job->id_slot  = id_slot;
    job->tokens   = tokens.clone();
    job->row_tgt  = row_tgt;
    job->row_dft  = row_dft;
    job->runs_tgt = runs_tgt;
    job->runs_dft = runs_dft;
    job->run_data = std::move(run_data);

    // the checkpoints up to its end, as persist() takes them, and the one at its end when the caller made one; a
    // checkpoint with bytes that is not the caller's to write counts as one handed back (named when stored)
    const int64_t n = (int64_t) tokens.size();
    std::vector<std::pair<disk_ckpt_ref, const common_prompt_checkpoint *>> refs;
    auto add = [&](const common_prompt_checkpoint & c, bool may_write) {
        if (c.n_tokens > n) {
            return;
        }
        disk_ckpt_ref r {};
        const common_prompt_checkpoint * data = nullptr;
        if (c.data_tgt.empty()) {
            const auto p = paged ? paged->find({ c.n_tokens, c.pos_min, c.pos_max }) : ckpt_paged_map::const_iterator();
            if (!paged || p == paged->end() ||
                ckpt_key(text, p->second.n_tokens, p->second.pos_min, p->second.pos_max,
                         p->second.size_tgt, p->second.size_dft, p->second.size_spec) != p->second.key) {
                return;
            }
            r = p->second;
        } else {
            r = { ckpt_key(text, c.n_tokens, c.pos_min, c.pos_max, c.data_tgt.size(), c.data_dft.size(), c.data_spec.size()),
                  c.n_tokens, c.pos_min, c.pos_max, c.data_tgt.size(), c.data_dft.size(), c.data_spec.size() };
            data = may_write ? &c : nullptr;
        }
        for (const auto & x : refs) {
            if (x.first.key == r.key) {
                return;
            }
        }
        refs.push_back({ r, data });
    };
    for (const auto & c : ckpts) {
        add(c, writable.count(&c) > 0);
    }
    if (end) {
        add(*end, true);
    }

    std::vector<bool> need(refs.size(), false);
    auto unpin_job = [&]() {
        std::lock_guard<std::mutex> lk(disk_mu);
        disk_unpin.insert(disk_unpin.end(), job->pinned.begin(), job->pinned.end());
        disk_unpin_runs.insert(disk_unpin_runs.end(), job->run_pinned.begin(), job->run_pinned.end());
        job->pinned.clear();
        job->run_pinned.clear();
        disk_cv.notify_all();
    };
    {
        std::lock_guard<std::mutex> lk(disk_mu);
        for (size_t i = 0; i < refs.size(); ++i) {
            auto o = disk_ckpts.find(refs[i].first.key);
            const bool stored = o != disk_ckpts.end() || disk_ckpt_in_flight(refs[i].first.key);
            if (refs[i].second) {
                need[i] = !stored;
                job->ckpts.push_back(refs[i].first);
            } else if (stored) {
                job->ckpts.push_back(refs[i].first);
            }
            if (!need[i] && o != disk_ckpts.end() && (refs[i].second || stored)) {
                o->second.refs++;
                job->pinned.push_back(refs[i].first.key);
            }
        }
        // the runs it names without their rows are in the store, pinned until the entry counts them, or on their way
        // there in a job ahead of this one. One that is neither went with an evicted entry: the slot starts over.
        std::set<std::pair<uint64_t, uint64_t>> carried;
        for (const auto & d : job->run_data) {
            carried.insert(d.first);
        }
        auto in_flight = [&](const std::pair<uint64_t, uint64_t> & k) {
            auto carries = [&](const disk_job & j) {
                return std::any_of(j.run_data.begin(), j.run_data.end(), [&](const auto & d) { return d.first == k; });
            };
            return std::any_of(disk_queue.begin(), disk_queue.end(), [&](const auto & j) { return carries(*j); }) ||
                   (disk_current && carries(*disk_current));
        };
        for (const auto * runs : { &job->runs_tgt, &job->runs_dft }) {
            for (const auto & r : *runs) {
                const std::pair<uint64_t, uint64_t> k { r.hi, r.lo };
                if (carried.count(k)) {
                    continue;
                }
                auto o = disk_runs.find(k);
                if (o != disk_runs.end()) {
                    o->second.refs++;
                    job->run_pinned.push_back(k);
                } else if (!in_flight(k)) {
                    disk_runs_lost.insert(id_slot);
                    disk_unpin.insert(disk_unpin.end(), job->pinned.begin(), job->pinned.end());
                    disk_unpin_runs.insert(disk_unpin_runs.end(), job->run_pinned.begin(), job->run_pinned.end());
                    disk_cv.notify_all();
                    return false;
                }
            }
        }
    }
    try {
        for (size_t i = 0; i < refs.size(); ++i) {
            if (need[i]) {
                job->ckpt_data.emplace_back(refs[i].first.key, *refs[i].second);
            }
        }
    } catch (const std::bad_alloc &) {
        SRV_WRN("%s", " - disk cache: not enough memory to copy the checkpoints, not saving\n");
        unpin_job();
        return false;
    }
    job->n_exact = runs_exact(job->tokens, job->runs_tgt, job->runs_dft, row_dft > 0, job->ckpts);

    std::unique_lock<std::mutex> lk(disk_mu);
    // a run is ~120 MB and a checkpoint ~110 MB: one job waiting besides the one being written, as for persist()
    if (!disk_queue.empty()) {
        if (!wait || disk_stop) {
            lk.unlock();
            unpin_job();
            return false;
        }
        disk_cv.wait(lk, [&] { return disk_queue.empty() || disk_stop; });
        if (disk_stop) {
            lk.unlock();
            unpin_job();
            return false;
        }
    }
    disk_queue.push_back(std::move(job));
    disk_cv.notify_all();
    return true;
}

uint64_t server_prompt_cache::ckpt_page_out(const server_tokens & tokens, std::list<common_prompt_checkpoint> & ckpts, ckpt_paged_map & paged) {
    if (disk_limit == 0) {
        return 0;
    }
    const llama_tokens text = tokens.get_text_tokens();
    if (text.size() != tokens.size()) {
        return 0;                                  // media: never stored
    }
    uint64_t freed = 0;
    std::lock_guard<std::mutex> lk(disk_mu);
    for (auto & c : ckpts) {
        if (c.data_tgt.empty()) {
            continue;                              // handed back already
        }
        const disk_ckpt_ref r { ckpt_key(text, c.n_tokens, c.pos_min, c.pos_max, c.data_tgt.size(), c.data_dft.size(), c.data_spec.size()),
                                c.n_tokens, c.pos_min, c.pos_max, c.data_tgt.size(), c.data_dft.size(), c.data_spec.size() };
        auto stored = disk_ckpts.find(r.key);
        if (stored == disk_ckpts.end()) {
            continue;                              // not stored, or only on its way: keep the bytes
        }
        // a pin: the store keeps the file while this slot may rewind to it, whatever becomes of its entry
        const std::tuple<int64_t, int32_t, int32_t> id { c.n_tokens, c.pos_min, c.pos_max };
        auto p = paged.find(id);
        if (p == paged.end() || p->second.key != r.key) {
            if (p != paged.end()) {
                disk_unpin.push_back(p->second.key);
            }
            stored->second.refs++;
        }
        paged[id] = r;
        freed += c.size();
        std::vector<uint8_t>().swap(c.data_tgt);
        std::vector<uint8_t>().swap(c.data_dft);
        std::vector<uint8_t>().swap(c.data_spec);
    }
    return freed;
}

bool server_prompt_cache::ckpt_page_in(const server_tokens & tokens, const ckpt_paged_map & paged, common_prompt_checkpoint & c) {
    const auto it = paged.find({ c.n_tokens, c.pos_min, c.pos_max });
    if (it == paged.end() || disk_limit == 0) {
        return false;
    }
    const disk_ckpt_ref & r = it->second;
    // the table outlives a change of conversation in the slot: the key says whether this is still the same one
    const llama_tokens text = tokens.get_text_tokens();
    if (ckpt_key(text, r.n_tokens, r.pos_min, r.pos_max, r.size_tgt, r.size_dft, r.size_spec) != r.key) {
        return false;
    }
    std::lock_guard<std::mutex> lk(disk_mu);       // the writer cannot delete it while it is read
    if (!disk_ckpts.count(r.key)) {
        return false;
    }
    common_prompt_checkpoint tmp;
    bool ok = false;
    try {
        spc_stage st;
        spc_file f(st);
        ok = read_ckpt(f, ckpt_path(disk_dir, r.key), r, tmp);
    } catch (const std::bad_alloc &) {
        return false;                              // no memory this time; the file is not to blame
    }
    if (!ok) {
        disk_bad_ckpts.push_back(r.key);           // retire it, and every entry that names it
        disk_cv.notify_all();
        return false;
    }
    c.data_tgt  = std::move(tmp.data_tgt);
    c.data_dft  = std::move(tmp.data_dft);
    c.data_spec = std::move(tmp.data_spec);
    return true;
}

void server_prompt_cache::ckpt_release(ckpt_paged_map & paged) {
    if (paged.empty()) {
        return;
    }
    std::lock_guard<std::mutex> lk(disk_mu);
    for (const auto & [id, r] : paged) {
        disk_unpin.push_back(r.key);
    }
    paged.clear();
    disk_cv.notify_all();
}

// caller holds disk_mu: pins let go of; an object nobody holds any more goes
void server_prompt_cache::disk_release_pins() {
    std::error_code ec;
    for (const uint64_t key : disk_unpin) {
        auto o = disk_ckpts.find(key);
        if (o != disk_ckpts.end() && --o->second.refs <= 0) {
            fs::remove(ckpt_path(disk_dir, key), ec);
            disk_bytes -= std::min(disk_bytes, o->second.bytes);
            disk_ckpts.erase(o);
        }
    }
    disk_unpin.clear();
    for (const auto & k : disk_unpin_runs) {
        auto o = disk_runs.find(k);
        if (o != disk_runs.end() && --o->second.refs <= 0) {
            fs::remove(run_path(disk_dir, k.first, k.second), ec);
            disk_bytes -= std::min(disk_bytes, o->second.bytes);
            disk_runs.erase(o);
        }
    }
    disk_unpin_runs.clear();
}

// caller holds disk_mu: objects a read found missing or damaged go, with every entry that names them
void server_prompt_cache::disk_retire_bad() {
    if (disk_bad_chunks.empty() && disk_bad_ckpts.empty() && disk_bad_runs.empty()) {
        return;
    }
    auto bad_chunk = [&](const disk_chunk_ref & c) {
        return std::find(disk_bad_chunks.begin(), disk_bad_chunks.end(), std::make_pair(c.hi, c.lo)) != disk_bad_chunks.end();
    };
    auto bad_ckpt = [&](const disk_ckpt_ref & k) {
        return std::find(disk_bad_ckpts.begin(), disk_bad_ckpts.end(), k.key) != disk_bad_ckpts.end();
    };
    auto bad_run = [&](const disk_run_ref & r) {
        return std::find(disk_bad_runs.begin(), disk_bad_runs.end(), std::make_pair(r.hi, r.lo)) != disk_bad_runs.end();
    };
    for (auto it = disk_index.begin(); it != disk_index.end();) {
        const bool names_bad = std::any_of(it->main.begin(), it->main.end(), bad_chunk)
                            || std::any_of(it->drft.begin(), it->drft.end(), bad_chunk)
                            || std::any_of(it->ckpts.begin(), it->ckpts.end(), bad_ckpt)
                            || std::any_of(it->runs_tgt.begin(), it->runs_tgt.end(), bad_run)
                            || std::any_of(it->runs_dft.begin(), it->runs_dft.end(), bad_run);
        if (names_bad) {
            SRV_WRN(" - disk cache: dropping %s: it names a damaged object\n", fs::path(it->path).filename().string().c_str());
            disk_doomed.push_back(std::move(*it));
            it = disk_index.erase(it);
        } else {
            ++it;
        }
    }
    std::error_code ec;
    for (const auto & k : disk_bad_chunks) {
        auto o = disk_chunks.find(k);
        if (o != disk_chunks.end()) {
            fs::remove(chunk_path(disk_dir, k.first, k.second), ec);
            disk_bytes -= std::min(disk_bytes, o->second.bytes);
            disk_chunks.erase(o);
        }
    }
    for (const uint64_t key : disk_bad_ckpts) {
        auto o = disk_ckpts.find(key);
        if (o != disk_ckpts.end()) {
            fs::remove(ckpt_path(disk_dir, key), ec);
            disk_bytes -= std::min(disk_bytes, o->second.bytes);
            disk_ckpts.erase(o);
        }
    }
    for (const auto & k : disk_bad_runs) {
        auto o = disk_runs.find(k);
        if (o != disk_runs.end()) {
            fs::remove(run_path(disk_dir, k.first, k.second), ec);
            disk_bytes -= std::min(disk_bytes, o->second.bytes);
            disk_runs.erase(o);
        }
    }
    disk_bad_chunks.clear();
    disk_bad_ckpts.clear();
    disk_bad_runs.clear();
}

void server_prompt_cache::disk_writer() {
    std::unique_lock<std::mutex> lk(disk_mu);
    // the main loop sleeps when every slot is idle; after each job it is woken once, so a save that found
    // the queue full is retried and the checkpoints that just landed can leave memory
    auto wake = [&]() {
        std::function<void()> cb = disk_on_written;
        lk.unlock();
        if (cb) {
            cb();
        }
        lk.lock();
    };
    while (true) {
        disk_retire_bad();
        disk_release_pins();
        // entries the main loop found unreadable: only this thread deletes
        while (!disk_doomed.empty()) {
            disk_drop(disk_doomed.back());
            disk_doomed.pop_back();
        }
        if (!disk_queue.empty()) {
            std::unique_ptr<disk_job> job = std::move(disk_queue.front());
            disk_queue.pop_front();
            disk_current = job.get();
            disk_cv.notify_all();                  // room in the queue
            lk.unlock();
            try {
                if (job->version == (int32_t) SPC_V3) {
                    disk_write_runs(*job);
                } else {
                    disk_write(*job);
                }
            } catch (const std::exception & ex) {
                SRV_ERR(" - disk cache: writing %d tokens failed: %s\n", (int) job->tokens.size(), ex.what());
            }
            lk.lock();
            disk_current = nullptr;
            disk_cv.notify_all();                  // a load may be waiting for exactly this entry
            lk.unlock();
            job.reset();                           // GBs: freed without the lock held
            lk.lock();
            wake();
            continue;
        }
        if (disk_stop) {
            break;                                 // saves drain first
        }
        disk_cv.wait(lk, [&] {
            return disk_stop || !disk_queue.empty() || !disk_doomed.empty() ||
                   !disk_bad_chunks.empty() || !disk_bad_ckpts.empty() || !disk_bad_runs.empty() ||
                   !disk_unpin.empty() || !disk_unpin_runs.empty();
        });
    }
}

// the writer thread, without disk_mu: new objects first (they have names nobody references yet, so
// nobody can be reading them), then the manifest, then - under the lock - the index
void server_prompt_cache::disk_write(disk_job & job) {
    const int64_t t0 = ggml_time_us();
    // the stored checkpoints the job names are pinned until its entry counts them; however this ends, they go
    // back (the writer's next round releases them)
    struct pins_back {
        server_prompt_cache & c;
        disk_job & j;
        ~pins_back() {
            std::lock_guard<std::mutex> lk(c.disk_mu);
            c.disk_unpin.insert(c.disk_unpin.end(), j.pinned.begin(), j.pinned.end());
            j.pinned.clear();
        }
    } release_on_exit { *this, job };
    {
        std::lock_guard<std::mutex> lk(disk_mu);
        if (disk_covered_locked(job.tokens, false) >= job.tokens.size()) {
            return;                                // on disk already, whole
        }
    }

    spc_stage st;
    disk_entry e;
    e.version   = (int32_t) SPC_V2;
    e.tokens    = job.tokens.clone();
    e.ckpts     = job.ckpts;
    e.main_size = job.main.size();
    e.drft_size = job.drft.size();

    std::set<std::pair<uint64_t, uint64_t>> fresh;         // chunks this job writes
    std::map<uint64_t, uint64_t>            fresh_ckpt;    // checkpoints this job writes -> bytes
    uint64_t wrote = 0, wrote_ckpt = 0, n_chunks = 0;
    int64_t t_cut = 0;
    const std::string path = manifest_path(disk_dir, e.tokens);

    // whatever this job wrote is unreferenced until the manifest is placed: on any failure, it goes
    auto discard = [&]() {
        std::error_code ec;
        fs::remove(path + ".part", ec);
        for (const auto & k : fresh) {
            fs::remove(chunk_path(disk_dir, k.first, k.second), ec);
        }
        for (const auto & [key, b] : fresh_ckpt) {
            fs::remove(ckpt_path(disk_dir, key), ec);
        }
    };

    bool ok = true;
    uint64_t mbytes = 0;
    try {
        auto store = [&](const std::vector<uint8_t> & v, std::vector<disk_chunk_ref> & refs) {
            for (size_t off = 0; ok && off < v.size();) {
                const int64_t c0 = ggml_time_us();
                const size_t n = cdc_next(v.data() + off, v.size() - off);
                const disk_chunk_ref c = chunk_ref(v.data() + off, n);
                t_cut += ggml_time_us() - c0;
                refs.push_back(c);
                ++n_chunks;
                bool have;
                {
                    std::lock_guard<std::mutex> lk(disk_mu);
                    have = disk_chunks.count({ c.hi, c.lo }) > 0;
                }
                if (!have && fresh.insert({ c.hi, c.lo }).second) {
                    const std::string p = chunk_path(disk_dir, c.hi, c.lo);
                    std::error_code ec;
                    fs::create_directories(fs::path(p).parent_path(), ec);
                    uint64_t b = 0;
                    ok = write_part(st, p + ".part", [&](spc_out & o) { return o.write(v.data() + off, n); }, b)
                      && place(p + ".part", p);
                    wrote += n;
                }
                off += n;
            }
        };
        store(job.main, e.main);
        store(job.drft, e.drft);

        for (const auto & [key, c] : job.ckpt_data) {
            if (!ok) {
                break;
            }
            bool have;
            {
                std::lock_guard<std::mutex> lk(disk_mu);
                have = disk_ckpts.count(key) > 0;
            }
            if (have || fresh_ckpt.count(key)) {
                continue;
            }
            const std::string p = ckpt_path(disk_dir, key);
            uint64_t b = 0;
            ok = write_part(st, p + ".part", [&](spc_out & o) { return write_ckpt_body(o, c); }, b) && place(p + ".part", p);
            if (ok) {
                fresh_ckpt[key] = b;
                wrote_ckpt += b;
            }
        }

        // a checkpoint this job names but did not carry was stored, or in flight, when the job was made; an
        // eviction or a failed job since may have taken it. Only this thread deletes, so what is present now
        // stays present until the index is updated below - the entry names exactly that.
        if (ok) {
            std::lock_guard<std::mutex> lk(disk_mu);
            const size_t before = e.ckpts.size();
            e.ckpts.erase(std::remove_if(e.ckpts.begin(), e.ckpts.end(), [&](const disk_ckpt_ref & k) {
                return !disk_ckpts.count(k.key) && !fresh_ckpt.count(k.key);
            }), e.ckpts.end());
            if (e.ckpts.size() != before) {
                SRV_WRN(" - disk cache: %zu checkpoints of %d tokens went while it was being written; saving it without them\n",
                        before - e.ckpts.size(), (int) e.tokens.size());
            }
        }
        ok = ok && write_manifest_part(st, path + ".part", e, disk_has_mtmd, mbytes);
    } catch (...) {
        discard();
        throw;
    }
    if (!ok) {
        SRV_WRN(" - disk cache: could not write %d tokens\n", (int) job.tokens.size());
        discard();
        return;
    }

    std::lock_guard<std::mutex> lk(disk_mu);
    if (!place(path + ".part", path)) {
        discard();
        return;
    }
    // the new entry's references first, so nothing it shares with an entry dropped below reaches zero
    for (const auto * refs : { &e.main, &e.drft }) {
        for (const auto & c : *refs) {
            auto & o = disk_chunks[{ c.hi, c.lo }];
            if (o.bytes == 0) {
                o.bytes = c.size;
                disk_bytes += c.size;
            }
            o.refs++;
        }
    }
    for (const auto & [key, b] : fresh_ckpt) {
        auto & o = disk_ckpts[key];
        o.bytes = b;
        disk_bytes += b;
    }
    for (const auto & k : e.ckpts) {
        disk_ckpts[k.key].refs++;
    }
    // an entry at the same path was just replaced by this manifest: its references go, its file stays
    for (auto it = disk_index.begin(); it != disk_index.end(); ++it) {
        if (it->path == path) {
            disk_drop(*it, /* remove_file = */ false);
            disk_index.erase(it);
            break;
        }
    }
    // entries this one extends are obsolete
    for (auto it = disk_index.begin(); it != disk_index.end();) {
        if (it->tokens.get_common_prefix(e.tokens) == it->tokens.size()) {
            disk_drop(*it);
            it = disk_index.erase(it);
        } else {
            ++it;
        }
    }
    e.path  = path;
    e.bytes = mbytes;
    e.order = ++disk_seq;
    disk_bytes += mbytes;
    const int n_tokens = (int) e.tokens.size();
    const size_t n_ck = e.ckpts.size();
    disk_index.push_back(std::move(e));
    disk_evict(path);

    const double dt = (ggml_time_us() - t0) / 1e6;
    const double gib = 1024.0 * 1024.0 * 1024.0;
    SRV_INF(" - disk cache: wrote %d tokens in %.0f ms, %.3f GiB in all: state %.3f of %.3f GiB (%zu of %llu chunks new), "
            "checkpoints %.3f GiB (%zu of %zu new); cutting %.0f ms - %zu entries, %.1f GiB on disk\n",
            n_tokens, dt * 1000.0, (wrote + wrote_ckpt) / gib, wrote / gib, (job.main.size() + job.drft.size()) / gib,
            fresh.size(), (unsigned long long) n_chunks, wrote_ckpt / gib, fresh_ckpt.size(), n_ck, t_cut / 1000.0,
            disk_index.size(), disk_bytes / gib);
}

// the writer thread, version 3: the runs this job carries (never one the store has), the checkpoints it carries,
// then the manifest naming all of them, then - under the lock - the index. A run the job names without its rows
// must still be in the store; if an eviction took it, the entry would have a hole, so the job goes and its slot
// starts its runs over.
void server_prompt_cache::disk_write_runs(disk_job & job) {
    const int64_t t0 = ggml_time_us();
    struct pins_back {
        server_prompt_cache & c;
        disk_job & j;
        ~pins_back() {
            std::lock_guard<std::mutex> lk(c.disk_mu);
            c.disk_unpin.insert(c.disk_unpin.end(), j.pinned.begin(), j.pinned.end());
            c.disk_unpin_runs.insert(c.disk_unpin_runs.end(), j.run_pinned.begin(), j.run_pinned.end());
            j.pinned.clear();
            j.run_pinned.clear();
        }
    } release_on_exit { *this, job };
    if (job.run_data.empty()) {
        std::lock_guard<std::mutex> lk(disk_mu);
        if ((int64_t) disk_covered_locked(job.tokens, false) >= job.n_exact) {
            return;                                // no new runs, and the store brings back as much already
        }
    }

    spc_stage st;
    disk_entry e;
    e.version  = (int32_t) SPC_V3;
    e.tokens   = job.tokens.clone();
    e.ckpts    = job.ckpts;
    e.row_tgt  = job.row_tgt;
    e.row_dft  = job.row_dft;
    e.runs_tgt = job.runs_tgt;
    e.runs_dft = job.runs_dft;

    std::map<std::pair<uint64_t, uint64_t>, uint64_t> fresh_run;   // runs this job writes -> file bytes
    std::map<uint64_t, uint64_t>                      fresh_ckpt;
    uint64_t wrote = 0, wrote_ckpt = 0;
    bool lost = false;
    const std::string path = manifest_path(disk_dir, e.tokens);

    auto discard = [&]() {
        std::error_code ec;
        fs::remove(path + ".part", ec);
        for (const auto & [k, b] : fresh_run) {
            fs::remove(run_path(disk_dir, k.first, k.second), ec);
        }
        for (const auto & [key, b] : fresh_ckpt) {
            fs::remove(ckpt_path(disk_dir, key), ec);
        }
    };

    bool ok = true;
    uint64_t mbytes = 0;
    try {
        for (const auto & [k, rows] : job.run_data) {
            if (!ok) {
                break;
            }
            bool have;
            {
                std::lock_guard<std::mutex> lk(disk_mu);
                have = disk_runs.count(k) > 0;
            }
            if (have || fresh_run.count(k)) {
                continue;
            }
            const disk_run_ref * r = nullptr;
            uint64_t row = 0;
            for (int b = 0; b < 2 && !r; ++b) {
                for (const auto & x : b == 0 ? e.runs_tgt : e.runs_dft) {
                    if (x.hi == k.first && x.lo == k.second) {
                        r   = &x;
                        row = b == 0 ? e.row_tgt : e.row_dft;
                        break;
                    }
                }
            }
            if (r == nullptr || rows.size() != row * (uint64_t) r->n) {
                SRV_ERR("%s", " - disk cache: a job's run does not match its list; not writing it\n");
                ok = false;
                break;
            }
            const std::string p = run_path(disk_dir, k.first, k.second);
            std::error_code ec;
            fs::create_directories(fs::path(p).parent_path(), ec);
            uint64_t b = 0;
            ok = write_part(st, p + ".part", [&](spc_out & o) { return write_run_body(o, row, *r, rows); }, b) && place(p + ".part", p);
            if (ok) {
                fresh_run[k] = b;
                wrote += b;
            }
        }

        for (const auto & [key, c] : job.ckpt_data) {
            if (!ok) {
                break;
            }
            bool have;
            {
                std::lock_guard<std::mutex> lk(disk_mu);
                have = disk_ckpts.count(key) > 0;
            }
            if (have || fresh_ckpt.count(key)) {
                continue;
            }
            const std::string p = ckpt_path(disk_dir, key);
            uint64_t b = 0;
            ok = write_part(st, p + ".part", [&](spc_out & o) { return write_ckpt_body(o, c); }, b) && place(p + ".part", p);
            if (ok) {
                fresh_ckpt[key] = b;
                wrote_ckpt += b;
            }
        }

        // only this thread deletes, so what is present now stays present until the index is updated below
        if (ok) {
            std::lock_guard<std::mutex> lk(disk_mu);
            for (const auto * runs : { &e.runs_tgt, &e.runs_dft }) {
                for (const auto & r : *runs) {
                    if (!disk_runs.count({ r.hi, r.lo }) && !fresh_run.count({ r.hi, r.lo })) {
                        lost = true;
                    }
                }
            }
            e.ckpts.erase(std::remove_if(e.ckpts.begin(), e.ckpts.end(), [&](const disk_ckpt_ref & k) {
                return !disk_ckpts.count(k.key) && !fresh_ckpt.count(k.key);
            }), e.ckpts.end());
            ok = !lost;
        }
        e.n_exact = runs_exact(e.tokens, e.runs_tgt, e.runs_dft, e.row_dft > 0, e.ckpts);
        ok = ok && write_manifest_part(st, path + ".part", e, disk_has_mtmd, mbytes);
    } catch (...) {
        discard();
        throw;
    }
    if (!ok) {
        discard();
        if (lost) {
            std::lock_guard<std::mutex> lk(disk_mu);
            disk_runs_lost.insert(job.id_slot);
            SRV_WRN(" - disk cache: runs of %d tokens went while they were being extended; slot %d writes them again\n",
                    (int) job.tokens.size(), job.id_slot);
        } else {
            SRV_WRN(" - disk cache: could not write %d tokens\n", (int) job.tokens.size());
        }
        return;
    }

    std::lock_guard<std::mutex> lk(disk_mu);
    if (!place(path + ".part", path)) {
        discard();
        return;
    }
    // the new entry's references first, so nothing it shares with an entry dropped below reaches zero
    for (const auto * runs : { &e.runs_tgt, &e.runs_dft }) {
        for (const auto & r : *runs) {
            auto & o = disk_runs[{ r.hi, r.lo }];
            const auto f = fresh_run.find({ r.hi, r.lo });
            if (o.bytes == 0 && f != fresh_run.end()) {
                o.bytes = f->second;
                disk_bytes += f->second;
            }
            o.refs++;
        }
    }
    for (const auto & [key, b] : fresh_ckpt) {
        auto & o = disk_ckpts[key];
        o.bytes = b;
        disk_bytes += b;
    }
    for (const auto & k : e.ckpts) {
        disk_ckpts[k.key].refs++;
    }
    for (auto it = disk_index.begin(); it != disk_index.end(); ++it) {
        if (it->path == path) {
            disk_drop(*it, /* remove_file = */ false);
            disk_index.erase(it);
            break;
        }
    }
    // an entry this one extends goes, if this one brings back at least as much: a conversation written through
    // leaves one entry, the latest, whose runs are all of the earlier ones' and more
    for (auto it = disk_index.begin(); it != disk_index.end();) {
        const int64_t theirs = it->version == (int32_t) SPC_V3 ? it->n_exact : (int64_t) it->tokens.size();
        if (it->tokens.get_common_prefix(e.tokens) == it->tokens.size() && e.n_exact >= theirs) {
            disk_drop(*it);
            it = disk_index.erase(it);
        } else {
            ++it;
        }
    }
    e.path  = path;
    e.bytes = mbytes;
    e.order = ++disk_seq;
    disk_bytes += mbytes;
    const int     n_tokens = (int) e.tokens.size();
    const int64_t n_exact  = e.n_exact;
    const size_t  n_runs   = e.runs_tgt.size() + e.runs_dft.size();
    disk_index.push_back(std::move(e));
    disk_evict(path);

    const double gib = 1024.0 * 1024.0 * 1024.0;
    SRV_INF(" - disk cache: wrote %d tokens (v3) in %.0f ms: %zu of %zu runs new (%.3f GiB), %zu checkpoints new (%.3f GiB); "
            "brings back %lld tokens - %zu entries, %.1f GiB on disk\n",
            n_tokens, (ggml_time_us() - t0) / 1000.0, fresh_run.size(), n_runs, wrote / gib, fresh_ckpt.size(), wrote_ckpt / gib,
            (long long) n_exact, disk_index.size(), disk_bytes / gib);
}

server_prompt_cache_state * server_prompt_cache::load_from_disk(const server_tokens & tokens_new, float & f_keep_best, float & f_sim_best,
                                                                ckpt_paged_map * paged) {
    std::unique_lock<std::mutex> lk(disk_mu);

    // an entry still on its way to disk may be exactly the one this prompt wants: let it land
    const int64_t tw0 = ggml_time_us();
    const float f_sim_base = f_sim_best;
    disk_cv.wait(lk, [&] { return !disk_in_flight(tokens_new, f_sim_base); });
    const double waited_ms = (ggml_time_us() - tw0) / 1000.0;

    auto best    = disk_index.end();
    auto closest = disk_index.end();          // closest entry whether or not it was taken, for the miss line
    int  closest_lcp = -1;
    int  best_lcp = -1;
    float best_keep = 0.0f, best_sim = 0.0f;
    for (auto it = disk_index.begin(); it != disk_index.end(); ++it) {
        const int lcp_cur = it->tokens.get_common_prefix(tokens_new);
        const float f_keep_cur = float(lcp_cur) / it->tokens.size();
        const float f_sim_cur  = float(lcp_cur) / tokens_new.size();
        if (lcp_cur > closest_lcp) { closest_lcp = lcp_cur; closest = it; }
        // version 3 is load_runs()'s, which found nothing in it to bring back when this runs
        if (it->version == (int32_t) SPC_V3) continue;
        // f_keep guards against dragging a long state in to serve a short prefix; f_sim has to beat
        // what the slot or the RAM tier already offers, or the read buys nothing
        if (f_keep_cur < 0.25f || f_sim_cur <= f_sim_best) continue;
        // among what is left, take the entry that skips the most tokens. Ranking by f_keep the way
        // the RAM tier does lets a short entry that happens to be a complete prefix (f_keep = 1.000)
        // shadow a much longer one for the same conversation, because nothing can then exceed it:
        // measured here as a 49370-token entry beating the 79441-token entry of the same chat and
        // costing 30071 tokens of prefill, about 40 s.
        if (lcp_cur > best_lcp) {
            best_lcp  = lcp_cur;
            best_keep = f_keep_cur;
            best_sim  = f_sim_cur;
            best      = it;
        }
    }
    if (best == disk_index.end()) {
        // the prefill that follows costs a minute or two on a long conversation, so name what was on
        // the shelf and how it lost: f_keep below 0.25 means the conversation forked away from the
        // entry, a base that already beats it means RAM or the slot covers more of the prompt
        if (closest != disk_index.end()) {
            SRV_INF(" - disk cache: nothing taken for %d tokens (%zu on disk; closest has %d tokens, lcp = %d, f_keep = %.3f, f_sim = %.3f; base f_keep = %.3f, f_sim = %.3f)\n",
                    (int) tokens_new.size(), disk_index.size(), (int) closest->tokens.size(), closest_lcp,
                    closest->tokens.size() ? float(closest_lcp) / closest->tokens.size() : 0.0f,
                    tokens_new.size() ? float(closest_lcp) / tokens_new.size() : 0.0f, f_keep_best, f_sim_best);
        } else {
            SRV_INF(" - disk cache: empty, %d tokens will be processed from scratch\n", (int) tokens_new.size());
        }
        return nullptr;
    }

    // the lock is held while the files are read: the writer cannot delete them underneath
    const int64_t t0 = ggml_time_us();
    g_spc_cost = {};
    server_prompt_cache_state state;
    std::string why;
    read_fault fault;
    bool ok = false;
    // an entry's checkpoints are objects of their own in the store: when the caller can page them in, they stay
    // there (a 79K-token conversation carries ~3.3 GB of them; a rewind reads one)
    const bool lazy = paged != nullptr;
    try {
        spc_stage stage;
        ok = read_entry(stage, disk_dir, *best, state, why, fault, lazy);
    } catch (const std::bad_alloc &) {
        SRV_WRN(" - disk cache: no memory to read %d tokens back, processing the prompt instead\n", (int) best->tokens.size());
        return nullptr;                       // the entry is fine; this time there was no room for it
    } catch (const std::exception & ex) {
        why = ex.what();
        ok = false;
    }
    if (!ok) {
        SRV_WRN(" - disk cache: %s is unreadable (%s), dropping it\n", fs::path(best->path).filename().string().c_str(), why.c_str());
        if (fault.chunk) {
            disk_bad_chunks.push_back({ fault.hi, fault.lo });
        }
        if (fault.ckpt) {
            disk_bad_ckpts.push_back(fault.key);
        }
        disk_doomed.push_back(std::move(*best));
        disk_index.erase(best);
        disk_cv.notify_all();
        return nullptr;
    }
    // the checkpoints left on disk are pinned, as a slot pins the ones it pages out: eviction of this entry must not
    // delete one the restored conversation may still rewind to. One the store no longer has is left out.
    size_t n_left = 0;
    if (lazy) {
        paged->clear();
        auto it = state.prompt.checkpoints.begin();
        for (const auto & k : best->ckpts) {
            GGML_ASSERT(it != state.prompt.checkpoints.end() && it->n_tokens == k.n_tokens);
            auto stored = disk_ckpts.find(k.key);
            if (stored == disk_ckpts.end()) {
                it = state.prompt.checkpoints.erase(it);
                continue;
            }
            stored->second.refs++;
            (*paged)[{ k.n_tokens, k.pos_min, k.pos_max }] = k;
            ++n_left;
            ++it;
        }
    }
    f_keep_best = best_keep;
    f_sim_best  = best_sim;
    best->order = ++disk_seq;
    std::error_code ec;
    fs::last_write_time(best->path, fs::file_time_type::clock::now(), ec);   // the LRU order survives a restart
    SRV_INF(" - disk cache: read %d tokens (v%d), %.3f GiB in %.0f ms (file %.0f ms at %.0f MB/s, buffers %.0f ms%s%s; f_keep = %.3f, f_sim = %.3f)\n",
            (int) state.prompt.tokens.size(), best->version, g_spc_cost.bytes / (1024.0 * 1024.0 * 1024.0), (ggml_time_us() - t0) / 1000.0,
            g_spc_cost.read_us / 1000.0, g_spc_cost.read_us ? g_spc_cost.bytes / 1048576.0 / (g_spc_cost.read_us / 1e6) : 0.0,
            g_spc_cost.alloc_us / 1000.0, waited_ms >= 1.0 ? (", waited " + std::to_string((int) waited_ms) + " ms for the writer").c_str() : "",
            lazy ? (", " + std::to_string(n_left) + " checkpoints left on disk").c_str() : "",
            f_keep_best, f_sim_best);
    states.push_back(std::move(state));
    return &states.back();
}

// version 3: pick the entry, pin what it names up to the restore point, let the lock go (making room may save a
// conversation, which takes it), then cells, rows a run at a time - the next run read while this one goes in - and
// the recurrent state from the checkpoint at the restore point. Two runs in memory at most, never the state.
bool server_prompt_cache::load_runs(server_prompt & prompt, const server_tokens & tokens_new, llama_context * ctx_tgt, llama_context * ctx_dft,
                                    int32_t id_slot, float & f_keep_best, float & f_sim_best, const std::function<void(size_t)> & before_restore,
                                    ckpt_paged_map * paged_out, disk_runs_state * runs_out, bool & restored, std::vector<uint8_t> * spec_out) {
    restored = false;
    const uint64_t row_tgt = llama_strix_kv_row_size(ctx_tgt);
    const uint64_t row_dft = ctx_dft ? llama_strix_kv_row_size(ctx_dft) : 0;
    if (disk_limit == 0 || row_tgt == 0 || (ctx_dft && row_dft == 0) || tokens_new.size() == 0 || paged_out == nullptr) {
        return false;
    }
    if (tokens_new.get_text_tokens().size() != tokens_new.size()) {
        return false;                              // media: never stored
    }

    server_tokens               tokens;            // the entry's, copied: the store can change once the lock goes
    std::string                 path;
    std::vector<disk_run_ref>   runs_tgt, runs_dft;
    std::vector<disk_ckpt_ref>  ckpts;             // the ones up to the restore point, pinned
    std::vector<std::pair<uint64_t, uint64_t>> run_pins;
    disk_ckpt_ref at {};
    int64_t n = 0;
    float keep = 0.0f;
    {
        std::unique_lock<std::mutex> lk(disk_mu);
        const float f_sim_base = f_sim_best;
        disk_cv.wait(lk, [&] { return !disk_in_flight(tokens_new, f_sim_base); });

        auto best = disk_index.end();
        for (auto it = disk_index.begin(); it != disk_index.end(); ++it) {
            if (it->version != (int32_t) SPC_V3 || it->row_tgt != row_tgt || (ctx_dft && it->row_dft != row_dft)) {
                continue;                          // another model's rows, or no draft rows for a server that drafts
            }
            const int64_t lcp = it->tokens.get_common_prefix(tokens_new);
            int64_t reach = std::min<int64_t>(lcp, runs_cover(it->runs_tgt));
            if (ctx_dft) {
                reach = std::min(reach, runs_cover(it->runs_dft));
            }
            const disk_ckpt_ref * k_best = nullptr;
            for (const auto & k : it->ckpts) {
                if (k.n_tokens > 0 && k.n_tokens <= reach && (!k_best || k.n_tokens > k_best->n_tokens) && disk_ckpts.count(k.key)) {
                    k_best = &k;
                }
            }
            if (!k_best) {
                continue;
            }
            // as for version 2: not a long entry dragged in for a short prefix, and it must beat what the slot or
            // the RAM tier already offers; among the rest, the one that skips the most tokens
            const float f_keep_cur = float(lcp) / it->tokens.size();
            const float f_sim_cur  = float(k_best->n_tokens) / tokens_new.size();
            if (f_keep_cur < 0.25f || f_sim_cur <= f_sim_best || k_best->n_tokens <= n) {
                continue;
            }
            n    = k_best->n_tokens;
            at   = *k_best;
            keep = f_keep_cur;
            best = it;
        }
        if (best == disk_index.end()) {
            return false;
        }
        tokens   = best->tokens.clone();
        path     = best->path;
        runs_tgt = best->runs_tgt;
        runs_dft = best->runs_dft;
        for (const auto & r : runs_tgt) {
            if (r.pos0 < n) {
                disk_runs[{ r.hi, r.lo }].refs++;
                run_pins.push_back({ r.hi, r.lo });
            }
        }
        if (ctx_dft) {
            for (const auto & r : runs_dft) {
                if (r.pos0 < n) {
                    disk_runs[{ r.hi, r.lo }].refs++;
                    run_pins.push_back({ r.hi, r.lo });
                }
            }
        }
        for (const auto & k : best->ckpts) {
            if (k.n_tokens <= n && disk_ckpts.count(k.key)) {
                disk_ckpts[k.key].refs++;
                ckpts.push_back(k);
            }
        }
        best->order = ++disk_seq;
        std::error_code ec;
        fs::last_write_time(best->path, fs::file_time_type::clock::now(), ec);   // the LRU order survives a restart
    }
    auto unpin = [&](bool ckpts_too) {
        std::lock_guard<std::mutex> lk(disk_mu);
        disk_unpin_runs.insert(disk_unpin_runs.end(), run_pins.begin(), run_pins.end());
        run_pins.clear();
        if (ckpts_too) {
            for (const auto & k : ckpts) {
                disk_unpin.push_back(k.key);
            }
            ckpts.clear();
        }
        disk_cv.notify_all();
    };

    const int64_t t0 = ggml_time_us();
    g_spc_cost = {};
    if (before_restore) {
        before_restore((size_t) n);
    }

    struct item {
        disk_run_ref    r;
        uint64_t        row;
        llama_context * ctx;
        int32_t         take;                      // positions of it that go in
    };
    std::vector<item> items;
    for (const auto & r : runs_tgt) {
        if (r.pos0 < n) {
            items.push_back({ r, row_tgt, ctx_tgt, (int32_t) std::min<int64_t>(r.n_use, n - r.pos0) });
        }
    }
    if (ctx_dft) {
        for (const auto & r : runs_dft) {
            if (r.pos0 < n) {
                items.push_back({ r, row_dft, ctx_dft, (int32_t) std::min<int64_t>(r.n_use, n - r.pos0) });
            }
        }
    }

    llama_tokens text = tokens.get_text_tokens();
    text.resize((size_t) n);
    std::string why;
    read_fault fault;
    bool ok = llama_strix_kv_alloc(ctx_tgt, id_slot, text.data(), (int32_t) n) &&
              (!ctx_dft || llama_strix_kv_alloc(ctx_dft, id_slot, text.data(), (int32_t) n));
    if (!ok) {
        why = "no cells for it";
    }
    size_t n_read = 0;
    try {
        spc_stage st[2];
        std::vector<uint8_t> buf[2];
        // a read runs on a pool thread: its cost comes back with it (the pool keeps its thread_locals between tasks)
        struct read_res { bool ok; spc_read_cost cost; };
        auto read_one = [&](size_t i) {
            const spc_read_cost c0 = g_spc_cost;
            spc_file f(st[i % 2]);
            const bool r = read_run(f, run_path(disk_dir, items[i].r.hi, items[i].r.lo), items[i].r, items[i].row, buf[i % 2]);
            return read_res { r, { g_spc_cost.alloc_us - c0.alloc_us, g_spc_cost.read_us - c0.read_us, g_spc_cost.bytes - c0.bytes } };
        };
        std::future<read_res> next;
        if (ok && !items.empty()) {
            next = std::async(std::launch::async, read_one, (size_t) 0);
        }
        for (size_t i = 0; ok && i < items.size(); ++i) {
            const read_res rr = next.get();
            g_spc_cost.alloc_us += rr.cost.alloc_us;
            g_spc_cost.read_us  += rr.cost.read_us;
            g_spc_cost.bytes    += rr.cost.bytes;
            if (!rr.ok) {
                ok = false;
                why = "missing or damaged run " + chunk_hex(items[i].r.hi, items[i].r.lo);
                fault.run = true;
                fault.hi  = items[i].r.hi;
                fault.lo  = items[i].r.lo;
                break;
            }
            if (i + 1 < items.size()) {
                next = std::async(std::launch::async, read_one, i + 1);
            }
            ok = llama_strix_kv_set_rows(items[i].ctx, id_slot, items[i].r.pos0, items[i].take, buf[i % 2].data(), items[i].r.n);
            if (!ok) {
                why = "the cache took no rows";
            }
            // STRIX_V3_VERIFY: read a whole run back out of the cache and check it against the hash it is named by
            if (ok && items[i].take == items[i].r.n && getenv("STRIX_V3_VERIFY")) {
                std::vector<uint8_t> back((size_t) items[i].row * (size_t) items[i].r.n);
                const bool got = llama_strix_kv_get_rows(items[i].ctx, id_slot, items[i].r.pos0, items[i].r.n, back.data(), back.size());
                const XXH128_hash_t h = XXH3_128bits(back.data(), back.size());
                if (!got || h.high64 != items[i].r.hi || h.low64 != items[i].r.lo) {
                    SRV_WRN(" - disk cache: VERIFY run at %d (%d rows, %s) reads back different%s\n", items[i].r.pos0, items[i].r.n,
                            items[i].ctx == ctx_tgt ? "target" : "draft", got ? "" : " (no rows)");
                }
            }
            ++n_read;
        }
        if (next.valid()) {
            next.wait();                           // a read still going when the loop stopped
        }
        if (ok) {
            common_prompt_checkpoint c;
            spc_file f(st[0]);
            if (!read_ckpt(f, ckpt_path(disk_dir, at.key), at, c)) {
                ok = false;
                why = "missing or damaged checkpoint";
                fault.ckpt = true;
                fault.key  = at.key;
            } else {
                ok = llama_state_seq_set_data_ext(ctx_tgt, c.data_tgt.data(), c.data_tgt.size(), id_slot,
                                                  LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == c.data_tgt.size() &&
                     (!ctx_dft || c.data_dft.empty() ||
                      llama_state_seq_set_data_ext(ctx_dft, c.data_dft.data(), c.data_dft.size(), id_slot,
                                                   LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == c.data_dft.size());
                if (!ok) {
                    why = "the recurrent state did not go in";
                } else if (spec_out) {
                    *spec_out = std::move(c.data_spec);   // the drafter's state at the restore point, when stored
                }
            }
        }
    } catch (const std::bad_alloc &) {
        ok = false;
        why = "no memory for a run";
    } catch (const std::exception & ex) {
        ok = false;
        why = ex.what();
    }

    if (!ok) {
        if (spec_out) {
            spec_out->clear();
        }
        // the cells were taken: give them back, and blame what was at fault
        llama_memory_seq_rm(llama_get_memory(ctx_tgt), id_slot, -1, -1);
        if (ctx_dft) {
            llama_memory_seq_rm(llama_get_memory(ctx_dft), id_slot, -1, -1);
        }
        {
            std::lock_guard<std::mutex> lk(disk_mu);
            if (fault.run) {
                disk_bad_runs.push_back({ fault.hi, fault.lo });
            }
            if (fault.ckpt) {
                disk_bad_ckpts.push_back(fault.key);
            }
        }
        unpin(true);
        SRV_WRN(" - disk cache: could not bring %lld tokens back from %s (%s); processing the prompt instead\n",
                (long long) n, fs::path(path).filename().string().c_str(), why.c_str());
        return true;                               // attempted: the slot's own conversation is gone
    }

    // the conversation up to the restore point, with its checkpoints left on disk and pinned for the slot
    prompt.tokens = tokens.clone();
    prompt.tokens.keep_first((size_t) n);
    prompt.checkpoints.clear();
    paged_out->clear();
    for (const auto & k : ckpts) {
        common_prompt_checkpoint c;
        c.n_tokens = k.n_tokens;
        c.pos_min  = k.pos_min;
        c.pos_max  = k.pos_max;
        prompt.checkpoints.push_back(std::move(c));
        (*paged_out)[{ k.n_tokens, k.pos_min, k.pos_max }] = k;
    }
    const size_t n_ckpts = ckpts.size();
    ckpts.clear();                                 // their pins went to paged_out
    if (runs_out) {
        auto cut = [n](const std::vector<disk_run_ref> & runs) {
            std::vector<disk_run_ref> res;
            for (auto r : runs) {
                if (r.pos0 >= n) {
                    break;
                }
                r.n_use = (int32_t) std::min<int64_t>(r.n_use, n - r.pos0);
                res.push_back(r);
            }
            return res;
        };
        runs_out->tgt    = cut(runs_tgt);
        runs_out->dft    = ctx_dft ? cut(runs_dft) : std::vector<disk_run_ref> {};
        runs_out->tokens = text;                   // the first n, which these runs were written for
    }
    unpin(false);

    f_keep_best = keep;
    f_sim_best  = float(n) / tokens_new.size();
    restored    = true;
    SRV_INF(" - disk cache: read %lld tokens (v3) into the cache, %.3f GiB in %.0f ms (file %.0f ms at %.0f MB/s; %zu runs, "
            "%zu checkpoints left on disk; f_keep = %.3f, f_sim = %.3f)\n",
            (long long) n, g_spc_cost.bytes / (1024.0 * 1024.0 * 1024.0), (ggml_time_us() - t0) / 1000.0, g_spc_cost.read_us / 1000.0,
            g_spc_cost.read_us ? g_spc_cost.bytes / 1048576.0 / (g_spc_cost.read_us / 1e6) : 0.0, n_read, n_ckpts, f_keep_best, f_sim_best);
    return true;
}
