#pragma once

#include "json-schema.h"
#include "json.h"

#include <functional>
#include <string>

std::string json_schema_to_grammar(const common_json & schema, bool force_gbnf = false);
std::string json_schema_to_grammar(const common_chat_schema_document & schema);

// strixllama: a tool's parameters schema with its properties and required list spelled out at the top level, however
// the schema states them: a top-level $ref (to #/$defs or #/definitions), allOf parts (each requiring what it requires),
// oneOf / anyOf alternatives (offered together, required only where every alternative requires them) and
// {not: {required: [...]}} (those left out). A schema that already lists its properties plainly comes back unchanged.
// For the formats that build one grammar rule per parameter; what the model is shown is not affected.
common_json common_tool_parameters_flatten(const common_json & params);

struct common_grammar_builder {
    std::function<std::string(const std::string &, const std::string &)>    add_rule;
    std::function<std::string(const std::string &, const common_chat_schema &)> add_schema;
};

struct common_grammar_options {
    bool dotall = false;
};

std::string gbnf_format_literal(const std::string & literal);

std::string build_grammar(const std::function<void(const common_grammar_builder &)> & cb, const common_grammar_options & options = {});
