"""请求样本回放校验：用真实 JSON 请求体分别校验旧版 / 新版操作的请求体 schema。

输入每条样本：旧版操作的 method + 路径模板（path template，如 /pets/{id}）、
Content-Type 与 JSON 请求体。新版按**相同 method 与路径模板**查找操作；
操作或媒体类型在新版消失即视为**新版拒绝**。

覆盖的 schema 约束（范围内）：type、required、enum、properties、items、
allOf 与本地 $ref（循环引用安全终止）。

范围外判定：
  - 无法解析的本地 $ref / 外部引用；
  - anyOf / oneOf / not（按需求只实现 allOf 合取）；
  - format / pattern / minimum / maximum / minLength / maxLength /
    minItems / maxItems / minProperties / maxProperties / uniqueItems 等
    本功能未实现的约束关键字；
校验中遇到上述构造时，该侧结果标为 **undetermined（无法判定）**，
不给出接受/拒绝结论、不计入风险。

风险定义：仅当 **旧版接受（accepted）而新版拒绝（rejected）** 时计为一条风险；
旧版拒绝的样本说明样本本身不合法，不计风险；任一侧无法判定也不计风险。

纯函数、无存储：结果只供本次查看，不改变审查结论。
"""
from __future__ import annotations

import json
from typing import Any

from .diff import HTTP_METHODS
from .refs import PointerError, normalize_pointer, resolve_pointer

_METHOD_set = {m.upper() for m in HTTP_METHODS}

# 范围内的校验关键字；其余 schema 约束关键字一律视为范围外 → 无法判定
_OUT_OF_SCOPE_KEYS = (
    "format", "pattern", "minimum", "maximum", "exclusiveMinimum",
    "exclusiveMaximum", "multipleOf", "minLength", "maxLength",
    "minItems", "maxItems", "uniqueItems", "contains", "minContains",
    "maxContains", "minProperties", "maxProperties", "patternProperties",
    "dependencies", "dependentRequired", "dependentSchemas",
    "unevaluatedProperties", "unevaluatedItems", "if", "then", "else",
    "propertyNames", "contentEncoding", "contentMediaType", "contentSchema",
)
_COMPOSITION_KEYS = ("anyOf", "oneOf")


class SampleRequestError(ValueError):
    """请求级格式错误（samples 缺失 / 非列表等），映射为 HTTP 400。"""


# ---------------------------------------------------------------------- #
# 操作与媒体类型定位
# ---------------------------------------------------------------------- #
def _deref_path_item(doc: dict[str, Any], raw_item: Any) -> Any:
    if isinstance(raw_item, dict) and isinstance(raw_item.get("$ref"), str):
        try:
            return resolve_pointer(doc, raw_item["$ref"])
        except PointerError:
            return raw_item
    return raw_item


def find_operation(doc: dict[str, Any], method: str, template: str) -> dict[str, Any] | None:
    """按 method（大写）+ 路径模板原样查找操作；不存在返回 None。"""
    paths = doc.get("paths")
    if not isinstance(paths, dict):
        return None
    item = _deref_path_item(doc, paths.get(template))
    if not isinstance(item, dict):
        return None
    op = item.get(method.lower())
    return op if isinstance(op, dict) else None


def _media_base(content_type: str) -> str:
    return (content_type or "").split(";", 1)[0].strip().lower()


def find_media_schema(
    doc: dict[str, Any], operation: dict[str, Any], content_type: str
) -> tuple[str, Any] | None:
    """在操作 requestBody.content 中按 Content-Type 查找媒体类型，返回其 schema。

    匹配规则：去参数后的精确匹配；支持 `类型/*` 与 `*/*` 通配，
    以及 `application/problem+json` 这类 `+json` 后缀（按 JSON 处理）。
    找不到（requestBody / content / 媒体类型消失）返回 None。
    """
    rb = operation.get("requestBody")
    try:
        rb = resolve_pointer(doc, rb["$ref"]) if isinstance(rb, dict) and isinstance(rb.get("$ref"), str) else rb
    except PointerError:
        return None
    if not isinstance(rb, dict) or not isinstance(rb.get("content"), dict):
        return None
    content = rb["content"]
    want = _media_base(content_type)
    if not want:
        return None

    if want in content:
        key = want
    else:
        main, _, sub = want.partition("/")
        json_like = sub == "json" or sub.endswith("+json")
        key = None
        for mt in content:
            base = _media_base(mt)
            if base == want:
                key = mt
                break
            b_main, _, b_sub = base.partition("/")
            if b_sub == "*" and (b_main == "*" or b_main == main):
                key = mt
                break
            # +json 结构化后缀只在 JSON 家族内互通（xml 不匹配 json）
            if json_like and b_sub.endswith("+json") and b_main == main:
                key = mt
                break
        if key is None:
            return None

    media = content[key]
    if not isinstance(media, dict) or "schema" not in media:
        # 媒体类型存在但未声明 schema：无约束（等价 true schema，接受任何值）
        return key, True
    # schema 可能是 dict / true / false / $ref 节点；返回值恒为元组
    return key, media["schema"]


# ---------------------------------------------------------------------- #
# Schema 校验（type / required / enum / properties / items / allOf / 本地 $ref）
# ---------------------------------------------------------------------- #
class _State:
    def __init__(self, doc: dict[str, Any]):
        self.doc = doc
        self.failures: list[dict[str, str]] = []
        self.notes: list[dict[str, str]] = []
        self._note_keys: set[tuple[str, str, str]] = set()

    def fail(self, path: str, code: str, message: str) -> None:
        self.failures.append({"path": path, "rule": code, "message": message})

    def note(self, path: str, code: str, message: str) -> None:
        key = (path, code, message)
        if key not in self._note_keys:
            self._note_keys.add(key)
            self.notes.append({"path": path, "rule": code, "message": message})


def _type_ok(declared: str, value: Any) -> bool:
    if declared == "string":
        return isinstance(value, str)
    if declared == "boolean":
        return isinstance(value, bool)
    if declared == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if declared == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if declared == "array":
        return isinstance(value, list)
    if declared == "object":
        return isinstance(value, dict)
    if declared == "null":
        return value is None
    return True


def _child_path(path: str, key: str) -> str:
    return f"{path}.{key}" if path else key


def _validate(
    st: _State,
    node: Any,
    value: Any,
    path: str,
    seen: frozenset[str],
    depth: int,
) -> None:
    """按合取（allOf）语义校验一个 schema 节点。

    合取成员通过 DFS 收集（$ref / allOf 展开），引用指针栈 `seen` 在整个
    收集过程中持续传递：A→B→（allOf）→A 形式的直接 / 间接循环引用都在
    再次遇到已在栈上的指针时安全终止。
    """
    if depth > 200:
        st.note(path, "depth-limit", "schema 嵌套过深（>200 层），超出部分无法判定。")
        return

    # ---- 收集合取成员（单引用 DFS，栈随引用进出） ----------------------- #
    members: list[dict[str, Any]] = []

    def collect(n: Any, stack: frozenset[str]) -> None:
        if n is True:
            return  # 布尔 schema true：无约束
        if n is False:
            members.append({"__bool_false__": True})
            return
        if isinstance(n, list):
            for item in n:
                collect(item, stack)
            return
        if not isinstance(n, dict):
            return

        ref = n.get("$ref")
        if isinstance(ref, str):
            try:
                pointer = normalize_pointer(ref)
            except PointerError:
                st.note(path, "unresolved-ref", f"遇到无法解析的 $ref：{ref}，该部分无法判定。")
                return
            if pointer in stack:
                return  # 循环引用：安全终止该分支
            try:
                target = resolve_pointer(st.doc, pointer)
            except PointerError:
                st.note(path, "unresolved-ref", f"遇到无法解析的 $ref：{ref}，该部分无法判定。")
                return
            next_stack = stack | {pointer}
            collect(target, next_stack)
            # OpenAPI 3.1：$ref 与兄弟字段并存，兄弟与引用目标按合取合并。
            # 把全部兄弟字段包成一个 schema 成员（required/type/allOf 等
            # 关键字在成员汇总阶段统一识别）。
            siblings = {k: v for k, v in n.items() if k != "$ref"}
            if siblings:
                collect(siblings, next_stack)
            return

        members.append(n)
        for sub in n.get("allOf") or []:
            collect(sub, stack)

    collect(node, seen)

    # ---- 合取语义下汇总约束 -------------------------------------------- #
    types: set[str] = set()
    nullable = False
    enums: list[list[Any]] = []
    required: set[str] = set()
    props: dict[str, Any] = {}
    items_schema: Any = None
    additional: Any = None  # None=未声明 / True / False / schema
    has_constraint = bool(members)

    for m in members:
        if m.get("__bool_false__"):
            st.fail(path, "schema-false", "schema 为 false（不允许任何值）。")
            continue

        t = m.get("type")
        if isinstance(t, str):
            types.add(t)
        elif isinstance(t, list):
            for x in t:
                if x == "null":
                    nullable = True
                elif isinstance(x, str):
                    types.add(x)
        if m.get("nullable") is True:  # OpenAPI 3.0
            nullable = True

        if isinstance(m.get("enum"), list):
            enums.append(m["enum"])
        if "const" in m:
            enums.append([m["const"]])

        for name in m.get("required") or []:
            if isinstance(name, str):
                required.add(name)

        mprops = m.get("properties")
        if isinstance(mprops, dict):
            for name, schema in mprops.items():
                if name in props and isinstance(schema, dict):
                    props[name] = {"allOf": [props[name], schema]}
                else:
                    props[name] = schema

        if isinstance(m.get("items"), dict):
            if items_schema is not None:
                items_schema = {"allOf": [items_schema, m["items"]]}
            else:
                items_schema = m["items"]

        if "additionalProperties" in m:
            ap = m["additionalProperties"]
            if ap is False:
                additional = False
            elif isinstance(ap, dict):
                additional = {"allOf": [additional, ap]} if isinstance(additional, dict) else ap
            elif ap is True and additional is None:
                additional = True

        for key in _OUT_OF_SCOPE_KEYS:
            if key in m:
                st.note(
                    path, "out-of-scope",
                    f"存在本功能范围外的约束关键字 '{key}'，其效果无法判定。",
                )
                break
        for key in _COMPOSITION_KEYS:
            if isinstance(m.get(key), list):
                st.note(
                    path, "out-of-scope",
                    f"存在范围外的组合关键字 '{key}'（本功能仅支持 allOf），其效果无法判定。",
                )
        if m.get("not") is not None:
            st.note(path, "out-of-scope", "存在范围外的 'not' 约束，其效果无法判定。")

    # ---- 应用约束 ------------------------------------------------------ #
    if not has_constraint:
        # 节点无法解析为任何有效 schema（损坏引用等已记 note）
        return

    # null：仅在允许可空时通过，且不再应用其它约束
    if value is None:
        if nullable:
            return
        if types and "null" not in types:
            st.fail(path, "type", f"类型应为 {sorted(types)}，实际为 null。")
            return
        if not types and (required or props or enums or items_schema is not None):
            st.fail(path, "type", "不允许 null 值。")
        return

    # type
    if types:
        if not any(_type_ok(t, value) for t in types):
            st.fail(
                path, "type",
                f"类型应为 {sorted(types)}，实际为 {_json_type(value)}。",
            )
            return  # 类型不符时后续结构校验无意义

    # enum（多重 enum 取交集）
    for allowed in enums:
        if not any(_enum_eq(value, v) for v in allowed):
            st.fail(
                path, "enum",
                f"值 {_short_repr(value)} 不在枚举允许值 {_short_repr(allowed)} 中。",
            )
            break

    # object: required / properties / additionalProperties
    if isinstance(value, dict):
        for name in sorted(required):
            if name not in value:
                st.fail(_child_path(path, name), "required", f"缺少必填字段 '{name}'。")
        for key, sub in value.items():
            sub_path = _child_path(path, str(key))
            if key in props:
                _validate(st, props[key], sub, sub_path, frozenset(), depth + 1)
            elif additional is False:
                st.fail(sub_path, "additional-properties", f"不允许额外字段 '{key}'。")
            elif isinstance(additional, dict):
                _validate(st, additional, sub, sub_path, frozenset(), depth + 1)

    # array: items
    if isinstance(value, list) and isinstance(items_schema, dict):
        for i, item in enumerate(value):
            _validate(st, items_schema, item, f"{path}[{i}]", frozenset(), depth + 1)


def _json_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def _enum_eq(a: Any, b: Any) -> bool:
    """枚举相等：严格区分 bool 与 int（JSON Schema 语义）。"""
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    return a == b


def _short_repr(value: Any, limit: int = 120) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        text = str(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def validate_body(doc: dict[str, Any], schema_node: Any, body: Any) -> dict[str, Any]:
    """对单个 JSON 请求体执行 schema 校验，返回侧结果。"""
    st = _State(doc)
    _validate(st, schema_node, body, "", frozenset(), 0)
    if st.failures:
        status = "rejected"
    elif st.notes:
        status = "undetermined"
    else:
        status = "accepted"
    return {
        "status": status,
        "failures": st.failures,
        "undetermined": st.notes,
    }


# ---------------------------------------------------------------------- #
# 逐条样本处理
# ---------------------------------------------------------------------- #
def _validate_sample(old_doc: dict[str, Any], new_doc: dict[str, Any], raw: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "index": None,
        "method": None,
        "path": None,
        "content_type": None,
        "error": None,
        "old": None,
        "new": None,
        "risk": None,
    }

    if not isinstance(raw, dict):
        result["error"] = "样本必须是包含 method/path/content_type/body 的对象。"
        return result

    result["index"] = raw.get("index")

    def _str_field(name: str, *, required_msg: str) -> tuple[str | None, str | None]:
        if name not in raw:
            return None, required_msg
        val = raw[name]
        if not isinstance(val, str):
            return None, f"{name} 必须是字符串，实际为 {_json_type(val)}。"
        text = val.strip()
        if not text:
            return None, f"{name} 为空。"
        return text, None

    method_text, err = _str_field("method", required_msg="缺少 method。")
    if err:
        result["error"] = err
        return result
    method = method_text.upper()

    template, err = _str_field("path", required_msg="缺少 path（旧版路径模板）。")
    if err:
        result["error"] = err
        return result

    ct_key = "content_type" if "content_type" in raw else (
        "contentType" if "contentType" in raw else "content_type"
    )
    content_type, err = _str_field(ct_key, required_msg="缺少 content_type。")
    if err:
        result["error"] = err
        return result

    result["method"] = method
    result["path"] = template
    result["content_type"] = content_type

    if method not in _METHOD_set:
        result["error"] = f"不支持的 HTTP 方法：{method}。"
        return result
    if not template.startswith("/"):
        result["error"] = f"路径模板必须以 / 开头：{template}。"
        return result

    raw_body = raw.get("body", "")
    body_present = "body" in raw
    if isinstance(raw_body, str):
        text: str | None = raw_body
    elif not body_present:
        text = ""
    elif raw_body is None:
        # 显式提交的 JSON null
        body = None
        text = None
    else:
        # 允许直接提交 JSON 对象（API 调用方便利）
        try:
            body = raw_body
            json.dumps(body)
            text = None
        except (TypeError, ValueError):
            result["error"] = "JSON 请求体无法序列化。"
            return result

    if text is not None:
        if not text.strip():
            result["error"] = "JSON 请求体为空（请提供 JSON 文本或对象；无请求体操作不适用本工具）。"
            return result
        try:
            body = json.loads(text)
        except json.JSONDecodeError as exc:
            result["error"] = f"JSON 请求体格式错误：第 {exc.lineno} 行第 {exc.colno} 列：{exc.msg}。"
            return result

    # ---- 旧版定位 ------------------------------------------------------ #
    old_op = find_operation(old_doc, method, template)
    if old_op is None:
        result["error"] = (
            f"旧版操作不存在：paths 中没有 {method} {template}，"
            "无法以旧版契约回放该样本。"
        )
        return result
    old_media = find_media_schema(old_doc, old_op, content_type)
    if old_media is None:
        result["error"] = (
            f"旧版操作 {method} {template} 的 requestBody 不接受媒体类型 "
            f"{content_type}（或该操作没有请求体），无法按旧版契约回放。"
        )
        return result
    old_mt, old_schema = old_media

    old_result = validate_body(old_doc, old_schema, body)
    old_result["media_type"] = old_mt
    result["old"] = old_result

    # 旧版拒绝的样本不合法，不再评估风险（新版仍照常回放给出参考信息）
    # ---- 新版定位（操作 / 媒体类型消失 = 新版拒绝） --------------------- #
    new_op = find_operation(new_doc, method, template)
    if new_op is None:
        result["new"] = {
            "status": "rejected",
            "media_type": None,
            "failures": [{
                "path": "",
                "rule": "operation-removed",
                "message": f"新版中操作 {method} {template} 已消失，旧版请求将被拒绝（404/405）。",
            }],
            "undetermined": [],
        }
    else:
        new_media = find_media_schema(new_doc, new_op, content_type)
        if new_media is None:
            result["new"] = {
                "status": "rejected",
                "media_type": None,
                "failures": [{
                    "path": "",
                    "rule": "media-type-removed",
                    "message": (
                        f"新版操作 {method} {template} 的 requestBody 不再接受"
                        f"媒体类型 {content_type}，旧版请求将被拒绝（415）。"
                    ),
                }],
                "undetermined": [],
            }
        else:
            new_mt, new_schema = new_media
            new_result = validate_body(new_doc, new_schema, body)
            new_result["media_type"] = new_mt
            result["new"] = new_result

    # ---- 风险判定 ------------------------------------------------------ #
    if result["old"]["status"] == "accepted":
        result["risk"] = result["new"]["status"] == "rejected"
    else:
        # 旧版拒绝 / 无法判定 → 不计风险
        result["risk"] = None
    return result


def build_sample_validation(
    old_doc: dict[str, Any],
    new_doc: dict[str, Any],
    samples: Any,
) -> dict[str, Any]:
    """逐条校验请求样本，返回逐条结果与风险汇总。纯函数、无存储。"""
    if not isinstance(samples, list):
        raise SampleRequestError("请求体须为 {\"samples\": [...]}，samples 为数组。")
    if not samples:
        raise SampleRequestError("samples 为空：至少提交一条请求样本。")
    if len(samples) > 200:
        raise SampleRequestError("samples 数量超过上限（200 条）。")

    items: list[dict[str, Any]] = []
    for i, raw in enumerate(samples):
        if isinstance(raw, dict):
            raw = {**raw, "index": raw.get("index", i + 1)}
        item = _validate_sample(old_doc, new_doc, raw)
        if not isinstance(item.get("index"), int):
            item["index"] = i + 1
        items.append(item)

    def _count(side: str, status: str) -> int:
        return sum(
            1 for it in items
            if not it["error"] and it[side] and it[side]["status"] == status
        )

    risks = [
        {
            "index": it["index"],
            "method": it["method"],
            "path": it["path"],
            "content_type": it["content_type"],
            "new_failures": it["new"]["failures"] if it["new"] else [],
        }
        for it in items
        if it.get("risk") is True
    ]

    summary = {
        "total": len(items),
        "errors": sum(1 for it in items if it["error"]),
        "old_accepted": _count("old", "accepted"),
        "old_rejected": _count("old", "rejected"),
        "old_undetermined": _count("old", "undetermined"),
        "new_accepted": _count("new", "accepted"),
        "new_rejected": _count("new", "rejected"),
        "new_undetermined": _count("new", "undetermined"),
        "risks": len(risks),
        "not_risk": sum(1 for it in items if it.get("risk") is False),
        "excluded": sum(1 for it in items if it.get("risk") is None and not it["error"]),
    }

    return {
        "summary": summary,
        "items": items,
        "risks": risks,
        "note": "回放结果只供本次查看，实时计算、不入库，不改变审查结论。",
    }
