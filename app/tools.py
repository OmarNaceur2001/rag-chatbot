"""Allow-listed assistant tools with validated arguments and confirmed file writes."""

import ast
import json
import math
import operator
import re
import secrets
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from app.common import BASE_DIR

TICKETS_FILE = BASE_DIR / "data" / "support_tickets.jsonl"
OUTBOX_FILE = BASE_DIR / "data" / "outbox.jsonl"


class StrictArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SearchDocumentsArgs(StrictArguments):
    query: str = Field(min_length=1, max_length=500)


class CreateTicketArgs(StrictArguments):
    subject: str = Field(min_length=3, max_length=120)
    description: str = Field(min_length=5, max_length=2000)


class SendEmailArgs(StrictArguments):
    recipient: str = Field(min_length=3, max_length=254)
    subject: str = Field(min_length=1, max_length=200)
    body: str = Field(min_length=1, max_length=4000)

    @field_validator("recipient")
    @classmethod
    def validate_recipient(cls, value: str) -> str:
        if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", value.strip()):
            raise ValueError("A valid email address is required.")
        return value.strip()


class CalculateArgs(StrictArguments):
    expression: str = Field(min_length=1, max_length=120)


ARGUMENT_MODELS: dict[str, type[StrictArguments]] = {
    "search_documents": SearchDocumentsArgs,
    "create_support_ticket": CreateTicketArgs,
    "send_email": SendEmailArgs,
    "calculate": CalculateArgs,
}

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": model.model_json_schema(),
            "strict": True,
        },
    }
    for name, description, model in (
        ("search_documents", "Search TechNova reference documents.", SearchDocumentsArgs),
        (
            "create_support_ticket",
            "Prepare a support ticket. The application will ask the user to confirm before saving it.",
            CreateTicketArgs,
        ),
        (
            "send_email",
            "Prepare an email for the demo outbox. The application will ask the user to confirm before queuing it.",
            SendEmailArgs,
        ),
        ("calculate", "Calculate a basic arithmetic expression.", CalculateArgs),
    )
]


@dataclass
class ToolOutcome:
    result: dict[str, Any]
    pending_action: dict[str, Any] | None = None


def _safe_calculate(expression: str) -> float | int:
    if not re.fullmatch(r"[0-9\s.+\-*/()%]+", expression):
        raise ValueError("Only basic arithmetic is supported.")
    tree = ast.parse(expression, mode="eval")
    operators: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
    }

    def evaluate(node: ast.AST) -> float | int:
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            value = node.value
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            operand = evaluate(node.operand)
            value = operand if isinstance(node.op, ast.UAdd) else -operand
        elif isinstance(node, ast.BinOp) and type(node.op) in operators:
            left = evaluate(node.left)
            right = evaluate(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 10:
                raise ValueError("Exponents must be between -10 and 10.")
            value = operators[type(node.op)](left, right)
        else:
            raise ValueError("The expression contains an unsupported operation.")
        if not math.isfinite(float(value)) or abs(value) > 1_000_000_000_000:
            raise ValueError("The calculated result is outside the supported range.")
        return value

    try:
        return evaluate(tree)
    except (ZeroDivisionError, OverflowError) as error:
        raise ValueError("The arithmetic expression is invalid or outside the supported range.") from error


def _prepare_pending_action(name: str, arguments: StrictArguments) -> dict[str, Any]:
    data = arguments.model_dump()
    if name == "create_support_ticket":
        summary = f"Create support ticket: {data['subject']}"
    else:
        summary = f"Queue email to {data['recipient']}: {data['subject']}"
    return {
        "action_id": secrets.token_urlsafe(18),
        "tool": name,
        "arguments": data,
        "summary": summary,
    }


def dispatch_tool(
    name: str,
    arguments_json: str,
    search_documents: Callable[[str], list[dict[str, Any]]],
) -> ToolOutcome:
    model = ARGUMENT_MODELS.get(name)
    if model is None:
        raise ValueError(f"Tool is not allow-listed: {name}")
    try:
        arguments = model.model_validate_json(arguments_json)
    except (ValidationError, ValueError):
        return ToolOutcome({"error": "Invalid tool arguments; check the required fields and types."})

    if name == "calculate":
        try:
            value = _safe_calculate(arguments.expression)
        except (SyntaxError, ValueError, ZeroDivisionError, OverflowError):
            return ToolOutcome({"error": "The arithmetic expression is invalid or unsupported."})
        return ToolOutcome({"result": value})

    if name == "search_documents":
        results = search_documents(arguments.query)
        return ToolOutcome({
            "results": [
                {
                    "text": item["text"][:1000],
                    "source": item["source"],
                    "page": item.get("page"),
                    "url": item.get("url"),
                }
                for item in results
            ]
        })

    return ToolOutcome(
        {"confirmation_required": True},
        pending_action=_prepare_pending_action(name, arguments),
    )


def execute_confirmed_action(pending_action: dict[str, Any]) -> dict[str, str]:
    name = pending_action.get("tool")
    model = ARGUMENT_MODELS.get(name)
    if name not in {"create_support_ticket", "send_email"} or model is None:
        raise ValueError("Pending action is not a supported write operation.")
    arguments = model.model_validate(pending_action.get("arguments"))
    now = datetime.now(timezone.utc).isoformat()

    if name == "create_support_ticket":
        ticket_id = f"TKT-{secrets.token_hex(4).upper()}"
        record = {
            "ticket_id": ticket_id,
            "created_at": now,
            **arguments.model_dump(),
        }
        _append_jsonl(TICKETS_FILE, record)
        return {"message": f"Ticket {ticket_id} was created."}

    recipient = arguments.recipient.strip()
    if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", recipient):
        raise ValueError("The email recipient is invalid.")
    _append_jsonl(OUTBOX_FILE, {"queued_at": now, **arguments.model_dump()})
    return {"message": f"Email to {recipient} was queued in the demo outbox."}


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")
