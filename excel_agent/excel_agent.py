"""
Excel Agent – an AI-powered assistant that can read, write, and analyze
Excel files using OpenAI function-calling (tool-use).

Usage
-----
    python excel_agent.py

Set OPENAI_API_KEY in your environment or in a .env file alongside this script.

The agent exposes the following tools to the LLM:
    - list_sheets        : list all sheet names in a workbook
    - read_sheet         : read a sheet and return it as a markdown table
    - get_summary_stats  : return descriptive statistics for numeric columns
    - write_sheet        : write/replace a sheet with provided row data
    - create_chart       : insert a bar chart into a sheet (saves the file)
"""

import json
import os
import sys

import openai
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ---------------------------------------------------------------------------
# Excel helpers
# ---------------------------------------------------------------------------

def list_sheets(file_path: str) -> dict:
    """Return a list of sheet names in the workbook."""
    try:
        xl = pd.ExcelFile(file_path)
        return {"sheets": xl.sheet_names}
    except Exception as exc:
        return {"error": str(exc)}


def read_sheet(file_path: str, sheet_name: str, max_rows: int = 50) -> dict:
    """Read *sheet_name* from *file_path* and return a markdown table."""
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name, nrows=max_rows)
        return {
            "rows": len(df),
            "columns": list(df.columns),
            "preview": df.to_markdown(index=False),
        }
    except Exception as exc:
        return {"error": str(exc)}


def get_summary_stats(file_path: str, sheet_name: str) -> dict:
    """Return descriptive statistics for numeric columns in *sheet_name*."""
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        stats = df.describe(include="number").to_dict()
        return {"stats": stats}
    except Exception as exc:
        return {"error": str(exc)}


def write_sheet(file_path: str, sheet_name: str, rows: list[dict]) -> dict:
    """
    Write *rows* (list of dicts) to *sheet_name* in *file_path*.
    Creates the file if it doesn't exist; replaces the sheet if it does.
    """
    try:
        df = pd.DataFrame(rows)
        mode = "a" if os.path.exists(file_path) else "w"
        kwargs = {"engine": "openpyxl", "mode": mode}
        if mode == "a":
            kwargs["if_sheet_exists"] = "replace"
        with pd.ExcelWriter(file_path, **kwargs) as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)
        return {"status": "ok", "file": file_path, "sheet": sheet_name, "rows_written": len(df)}
    except Exception as exc:
        return {"error": str(exc)}


def create_chart(file_path: str, sheet_name: str, title: str = "Chart") -> dict:
    """
    Insert a simple bar chart into *sheet_name* based on numeric columns.
    The first column is used as categories; remaining numeric columns as series.
    """
    try:
        from openpyxl import load_workbook
        from openpyxl.chart import BarChart, Reference

        wb = load_workbook(file_path)
        if sheet_name not in wb.sheetnames:
            return {"error": f"Sheet '{sheet_name}' not found"}
        ws = wb[sheet_name]

        max_row = ws.max_row
        max_col = ws.max_column

        chart = BarChart()
        chart.type = "col"
        chart.title = title
        chart.y_axis.title = "Value"
        chart.x_axis.title = ws.cell(1, 1).value or "Category"

        data = Reference(ws, min_col=2, max_col=max_col, min_row=1, max_row=max_row)
        cats = Reference(ws, min_col=1, min_row=2, max_row=max_row)
        chart.add_data(data, titles_from_data=True)
        chart.set_categories(cats)

        ws.add_chart(chart, f"A{max_row + 2}")
        wb.save(file_path)
        return {"status": "ok", "file": file_path, "sheet": sheet_name}
    except Exception as exc:
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Tool definitions for OpenAI
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_sheets",
            "description": "List all sheet names in an Excel workbook.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to the Excel file (.xlsx)"},
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_sheet",
            "description": "Read a sheet from an Excel file and return its contents as a markdown table.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to the Excel file"},
                    "sheet_name": {"type": "string", "description": "Name of the sheet to read"},
                    "max_rows": {
                        "type": "integer",
                        "description": "Maximum number of rows to return (default 50)",
                        "default": 50,
                    },
                },
                "required": ["file_path", "sheet_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_summary_stats",
            "description": "Return descriptive statistics (count, mean, std, min, max, percentiles) for numeric columns in a sheet.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to the Excel file"},
                    "sheet_name": {"type": "string", "description": "Name of the sheet"},
                },
                "required": ["file_path", "sheet_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_sheet",
            "description": "Write rows of data to a sheet in an Excel file. Creates the file and/or sheet if needed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to the Excel file"},
                    "sheet_name": {"type": "string", "description": "Name of the sheet to write"},
                    "rows": {
                        "type": "array",
                        "description": "List of objects (one per row). Keys become column headers.",
                        "items": {"type": "object"},
                    },
                },
                "required": ["file_path", "sheet_name", "rows"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_chart",
            "description": "Insert a bar chart into a sheet, using the first column as categories and remaining numeric columns as data series.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to the Excel file"},
                    "sheet_name": {"type": "string", "description": "Sheet to add the chart to"},
                    "title": {"type": "string", "description": "Chart title", "default": "Chart"},
                },
                "required": ["file_path", "sheet_name"],
            },
        },
    },
]

TOOL_FUNCTIONS = {
    "list_sheets": list_sheets,
    "read_sheet": read_sheet,
    "get_summary_stats": get_summary_stats,
    "write_sheet": write_sheet,
    "create_chart": create_chart,
}

# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an Excel Agent.
You can read, write, and analyze Excel (.xlsx) files using the tools provided.
When a user asks you to perform a task on an Excel file, call the appropriate tool(s).
Always confirm what you did and summarize key findings after using a tool."""


def run_agent(user_message: str, model: str = "gpt-4o") -> str:
    """Run the Excel agent for a single user message and return the final reply."""
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    while True:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )
        message = response.choices[0].message

        if message.tool_calls:
            messages.append(message)
            for tool_call in message.tool_calls:
                fn_name = tool_call.function.name
                fn_args = json.loads(tool_call.function.arguments)
                fn = TOOL_FUNCTIONS.get(fn_name)
                if fn:
                    result = fn(**fn_args)
                else:
                    result = {"error": f"Unknown tool: {fn_name}"}
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result),
                    }
                )
        else:
            return message.content or ""


# ---------------------------------------------------------------------------
# Interactive CLI
# ---------------------------------------------------------------------------

def main() -> None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable is not set.", file=sys.stderr)
        print("Set it in your environment or create a .env file with OPENAI_API_KEY=<your-key>", file=sys.stderr)
        sys.exit(1)

    print("Excel Agent – type your request and press Enter. Type 'exit' or 'quit' to stop.\n")
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        reply = run_agent(user_input)
        print(f"\nAgent: {reply}\n")


if __name__ == "__main__":
    main()
