"""
Condo Agent: Autonomous Condominium Emergency Coordinator & Auto-Dispatch Bot.
Powered by Google Gemini + python-telegram-bot + SQLite Central Database.
"""

import asyncio
import html
import io
import logging
import os
import random
import re
import sys
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from telegram import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Update,
)
from telegram.constants import ChatAction, ParseMode
from telegram.ext import (
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from condo_classifier.backends import BaselineBackend
from condo_classifier.database import (
    EVIDENCE_DIR,
    create_ticket,
    get_ticket,
    init_db,
    list_tickets,
    record_resident_feedback,
)
from condo_classifier.dispatch import (
    load_vendor_directory,
    resolve_vendor_assignment,
    trigger_auto_dispatch,
)
from condo_classifier.graph_rag import get_estate_vault
from condo_classifier.schema import ResidentRequest
from condo_classifier.service import Classifier
from condo_classifier.signals import safety_signal

local_classifier = Classifier(BaselineBackend())

# Load environment variables
load_dotenv()

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
# Preferred models in order of stability and throughput
GEMINI_MODELS = [
    os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-flash-latest",
]
GEMINI_MODEL = GEMINI_MODELS[0]

# Initialize Gemini Client
genai_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None


# Structured response schema from Gemini
class TriageDecision(BaseModel):
    category: str = Field(
        description="One of: maintenance, security, noise, cleanliness, facilities, "
        "parking, billing, general_enquiry, other"
    )
    priority: str = Field(description="One of: emergency, high, normal, low")
    issue_summary: str = Field(description="Clear 1-sentence summary of the incident")
    unit_number: str | None = Field(
        default=None, description="Resident unit number if mentioned (e.g. #08-14)"
    )
    is_emergency: bool = Field(
        description="True if pipe burst, major leak, lift entrapment, electrical fire, or intruder"
    )
    assigned_entity: str = Field(description="Name of the contractor or internal team assigned")
    entity_contact: str = Field(description="Phone or contact number of the assigned entity")
    eta_minutes: int = Field(
        description="Estimated arrival time in minutes (15-30 for emergencies, 120+ for normal)"
    )
    safety_instruction: str | None = Field(
        default=None,
        description="Urgent interim advice (e.g. turn off water valve, do not use lift)",
    )


def build_triage_system_prompt() -> str:
    """Builds the triage system prompt dynamically from vendors.json."""
    directory = load_vendor_directory()
    vendor_lines = []
    for cat, vendors in directory.items():
        if isinstance(vendors, dict):
            for sub, v in vendors.items():
                if isinstance(v, dict) and "vendor_name" in v:
                    phone = v.get("emergency_phone") or v.get("phone", "N/A")
                    sla = v.get("emergency_sla_minutes") or v.get("routine_sla_minutes", 30)
                    vendor_lines.append(
                        f"- {cat.title()} ({sub}): '{v['vendor_name']}' "
                        f"(Phone: {phone}, SLA: ~{sla} mins)"
                    )
    v_text = "\n".join(vendor_lines) if vendor_lines else "- Default Contractor (+65 6000-0000)"

    return (
        "You are the AI Emergency Coordinator and Triage Agent for Oceanfront Luxury Condominium.\n"
        "Assess resident requests (text/photos), classify incidents, and assign vendors.\n\n"
        f"VENDOR DIRECTORY (Loaded dynamically from vendors.json):\n{v_text}\n\n"
        "RULES:\n"
        "1. Active water leaks, burst pipes, stuck lifts, fire hazards are EMERGENCY.\n"
        "2. Provide actionable safety instructions when water or lift faults occur.\n"
        "3. If resident includes unit number (e.g. #08-14), capture it accurately.\n"
    )


def generate_work_order_id() -> str:
    return f"WO-{time.strftime('%Y%m%d')}-{random.randint(1000, 9999)}"


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    welcome_text = (
        "🏢 <b>Welcome to Oceanfront Residences Help Desk</b>\n\n"
        "I am your 24/7 AI Property Coordinator. I help you report issues, request services, "
        "and automatically dispatch on-call contractors for emergencies.\n\n"
        "💡 <b>Commands & Usage:</b>\n"
        "• Send a message describing your problem "
        "(e.g., <i>'Water pipe burst in kitchen #08-14!'</i>)\n"
        "• Or send a <b>photo</b> of the issue with a caption\n"
        "• <code>/ask &lt;QUESTION&gt;</code> - Ask estate by-laws & risers (Graph RAG)\n"
        "• <code>/status &lt;TICKET_ID&gt;</code> - Check status of a work order\n"
        "• <code>/mytickets</code> - View your active tickets\n\n"
        "🚨 <i>For urgent emergencies (burst pipes, lift faults), help is auto-dispatched.</i>"
    )
    if update.message:
        await update.message.reply_text(welcome_text, parse_mode=ParseMode.HTML)


async def ask_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Query the Condo Estate Knowledge Graph RAG directly."""
    message = update.message
    if not message:
        return

    query_text = " ".join(context.args) if context.args else ""
    if not query_text:
        await message.reply_text(
            "🏛️ <b>Estate Knowledge Graph Assistant</b>\n\n"
            "Ask any question about estate by-laws, risers, facilities, or policies.\n\n"
            "<b>Examples:</b>\n"
            "• <code>/ask Can my contractor drill tomorrow Saturday?</code>\n"
            "• <code>/ask What is the deposit to book BBQ Pit 1?</code>\n"
            "• <code>/ask Water leak in master bath ceiling #08-04</code>\n"
            "• <code>/ask What is the wheel clamp release fee?</code>",
            parse_mode=ParseMode.HTML,
        )
        return

    await message.chat.send_action(ChatAction.TYPING)
    vault = get_estate_vault()
    result = vault.query(query_text, hops=2, max_notes=12)

    # Format cleanly for Telegram HTML
    safe_answer = html.escape(result.answer)
    formatted_body = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", safe_answer)
    formatted_body = re.sub(r"### (.*?)\n", r"\n<b>\1</b>\n", formatted_body)

    response_text = (
        f"🏛️ <b>Estate Knowledge Graph Intelligence</b>\n\n"
        f"{formatted_body}\n\n"
        f"<i>🔍 Subgraph Context: {len(result.expanded_nodes)} connected entities traversed</i>"
    )
    await message.reply_text(response_text, parse_mode=ParseMode.HTML)


async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message
    if not message:
        return

    args = context.args or []
    if not args:
        await message.reply_text(
            "Please provide a ticket ID. Example: <code>/status WO-20261014-1001</code>",
            parse_mode=ParseMode.HTML,
        )
        return

    ticket_id = args[0].strip().upper()
    ticket = get_ticket(ticket_id)
    if not ticket:
        await message.reply_text(
            f"❌ Ticket <code>{html.escape(ticket_id)}</code> not found.",
            parse_mode=ParseMode.HTML,
        )
        return

    status_emoji = {
        "SUBMITTED": "📋",
        "DISPATCHED": "🚀",
        "IN_PROGRESS": "🚗",
        "RESOLVED": "✅",
    }.get(ticket["status"], "ℹ️")

    text = (
        f"📋 <b>Work Order Status:</b> <code>{html.escape(ticket['ticket_id'])}</code>\n\n"
        f"• <b>Status:</b> {status_emoji} <b>{html.escape(ticket['status'])}</b>\n"
        f"• <b>Category:</b> {html.escape(ticket['category'].title())}\n"
        f"• <b>Priority:</b> {html.escape(ticket['priority'].upper())}\n"
        f"• <b>Summary:</b> {html.escape(ticket['issue_summary'])}\n"
        f"• <b>Assigned Entity:</b> {html.escape(ticket.get('assigned_entity') or 'Unassigned')}\n"
        f"• <b>Created:</b> {html.escape(ticket['created_at'][:19].replace('T', ' '))}\n"
    )
    await message.reply_text(text, parse_mode=ParseMode.HTML)


async def mytickets_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message
    if not message or not message.chat:
        return

    chat_id = message.chat.id
    # Fetch tickets matching this telegram_chat_id or all recent if chat id filter
    all_tickets = list_tickets(limit=50)
    user_tickets = [t for t in all_tickets if t.get("telegram_chat_id") == chat_id]

    if not user_tickets:
        await message.reply_text(
            "📋 <b>You currently have no active work orders submitted from this chat.</b>\n\n"
            "To report an issue or emergency, send a message or photo describing the problem.",
            parse_mode=ParseMode.HTML,
        )
        return

    body = "📋 <b>Your Recent Condominium Tickets:</b>\n\n"
    for t in user_tickets[:5]:
        body += (
            f"• <code>{html.escape(t['ticket_id'])}</code> | "
            f"<b>{html.escape(t['status'])}</b>\n"
            f"  {html.escape(t['issue_summary'][:50])}...\n"
        )
    body += "\nUse <code>/status &lt;TICKET_ID&gt;</code> for details."
    await message.reply_text(body, parse_mode=ParseMode.HTML)


async def process_triage(text: str, photo_bytes: bytes | None = None) -> TriageDecision:
    def fallback_triage() -> TriageDecision:
        # Fallback deterministic safety signals and vendor routing
        safety = safety_signal(text)
        is_emergency = safety is not None
        try:
            classification = local_classifier.classify(ResidentRequest(text=text))
            category_str = classification.category.value
            priority_str = "emergency" if is_emergency else classification.priority.value
        except Exception:
            category_str = "other"
            priority_str = "emergency" if is_emergency else "normal"

        vendor = resolve_vendor_assignment(category_str, text)
        vendor_name = vendor.get("vendor_name", "Condo Concierge")
        contact = vendor.get("emergency_phone" if is_emergency else "phone", "+65 6111-5555")
        sla = vendor.get("emergency_sla_minutes" if is_emergency else "routine_sla_minutes", 120)

        unit_match = re.search(r"#\d{1,2}-\d{1,4}", text)
        unit_number = unit_match.group(0) if unit_match else None

        return TriageDecision(
            category=category_str,
            priority=priority_str,
            issue_summary=text[:100],
            unit_number=unit_number,
            is_emergency=is_emergency,
            assigned_entity=vendor_name,
            entity_contact=contact,
            eta_minutes=sla,
            safety_instruction=safety.evidence if safety else None,
        )

    if not genai_client:
        return fallback_triage()

    contents = []
    if photo_bytes:
        contents.append(types.Part.from_bytes(data=photo_bytes, mime_type="image/jpeg"))
    contents.append(text)

    # Try candidate models with graceful degradation
    tried_models = []
    for model_name in dict.fromkeys(GEMINI_MODELS):
        tried_models.append(model_name)
        try:
            response = await asyncio.to_thread(
                genai_client.models.generate_content,
                model=model_name,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=build_triage_system_prompt(),
                    response_mime_type="application/json",
                    response_schema=TriageDecision,
                    temperature=0.1,
                ),
            )
            return TriageDecision.model_validate_json(response.text)
        except Exception as exc:
            logger.warning(f"Model {model_name} failed ({exc}). Trying next candidate...")

    logger.warning(f"All Gemini models {tried_models} unavailable. Falling back to local triage.")
    return fallback_triage()


async def handle_resident_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message
    if not message:
        return

    text = message.text or message.caption or ""
    photo_bytes = None
    saved_photo_path = None

    if message.photo:
        await message.chat.send_action(ChatAction.UPLOAD_PHOTO)
        photo_file = await message.photo[-1].get_file()
        photo_stream = io.BytesIO()
        await photo_file.download_to_memory(photo_stream)
        photo_bytes = photo_stream.getvalue()
        if not text:
            text = "Please examine this photo of an issue reported in my condominium unit."

    await message.chat.send_action(ChatAction.TYPING)

    try:
        decision = await process_triage(text, photo_bytes)
        work_order = generate_work_order_id()

        # Save photo attachment if present
        if photo_bytes:
            EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
            photo_dest = EVIDENCE_DIR / f"{work_order}.jpg"
            photo_dest.write_bytes(photo_bytes)
            saved_photo_path = str(photo_dest)

        # Save ticket to Central SQLite Database
        chat_id = message.chat_id if message.chat else None
        resident_user = message.from_user.full_name if message.from_user else "Resident"
        ticket_status = "DISPATCHED" if decision.is_emergency else "SUBMITTED"

        create_ticket(
            ticket_id=work_order,
            source="TELEGRAM",
            category=decision.category,
            priority=decision.priority,
            issue_summary=decision.issue_summary,
            raw_message=text,
            unit_number=decision.unit_number,
            resident_name=resident_user,
            telegram_chat_id=chat_id,
            assigned_entity=decision.assigned_entity,
            entity_contact=decision.entity_contact,
            eta_minutes=decision.eta_minutes,
            safety_instruction=decision.safety_instruction,
            photo_path=saved_photo_path,
            status=ticket_status,
        )

        # Trigger automated contractor dispatch and guardhouse clearance if emergency
        guardhouse_pass = None
        if decision.is_emergency:
            assigned_dict = {
                "vendor_name": decision.assigned_entity,
                "phone": decision.entity_contact,
                "emergency_phone": decision.entity_contact,
                "emergency_sla_minutes": decision.eta_minutes,
                "routine_sla_minutes": decision.eta_minutes,
            }
            dispatch_res = trigger_auto_dispatch(
                ticket_id=work_order,
                category=decision.category,
                priority=decision.priority,
                issue_summary=decision.issue_summary,
                raw_message=text,
                unit_number=decision.unit_number,
                assigned_vendor=assigned_dict,
            )
            guardhouse_pass = dispatch_res.get("passcode")

        # Build response message with safe HTML escaping
        priority_emoji = {
            "emergency": "🚨 <b>CRITICAL EMERGENCY</b>",
            "high": "⚠️ <b>HIGH PRIORITY</b>",
            "normal": "📋 <b>NORMAL REQUEST</b>",
            "low": "ℹ️ <b>INFORMATIONAL</b>",
        }.get(decision.priority.lower(), "📋")

        status_header = (
            "🚀 <b>AUTOMATIC DISPATCH CONFIRMED</b>"
            if decision.is_emergency
            else "✅ <b>TICKET RECEIVED & QUEUED</b>"
        )

        safe_summary = html.escape(decision.issue_summary)
        safe_entity = html.escape(decision.assigned_entity)
        safe_contact = html.escape(decision.entity_contact)
        safe_category = html.escape(decision.category.title())

        reply_body = (
            f"{status_header}\n\n"
            f"<b>Ticket:</b> <code>{html.escape(work_order)}</code>\n"
            f"<b>Status:</b> {priority_emoji}\n"
            f"<b>Category:</b> {safe_category}\n"
            f"<b>Summary:</b> {safe_summary}\n"
        )

        if decision.unit_number:
            reply_body += f"<b>Unit:</b> <code>{html.escape(decision.unit_number)}</code>\n"

        if decision.is_emergency and guardhouse_pass:
            safe_pass = html.escape(guardhouse_pass)
            reply_body += f"🛡️ <b>Guardhouse Clearance:</b> <code>{safe_pass}</code> (Pre-cleared)\n"

        reply_body += (
            f"\n🔧 <b>Assigned Entity:</b> {safe_entity}\n"
            f"📞 <b>Contact:</b> <code>{safe_contact}</code>\n"
            f"⏱️ <b>Estimated Arrival / SLA:</b> ~{decision.eta_minutes} minutes\n"
        )

        if decision.safety_instruction:
            safe_instruction = html.escape(decision.safety_instruction)
            reply_body += f"\n💡 <b>Immediate Resident Guidance:</b>\n<i>{safe_instruction}</i>\n"

        # Enrich with Estate Knowledge Graph RAG (Valves, Risers, By-Laws & Policy Answers)
        estate_rag = get_estate_vault().query(text, unit_hint=decision.unit_number, hops=2)
        if estate_rag.grounded:
            # If the user asked a question or it is an inquiry, include the direct policy answer
            is_inquiry = decision.priority.lower() == "low" or decision.category.lower() in [
                "general_enquiry",
                "other",
                "facilities",
                "noise",
                "renovation",
            ] or "?" in text
            if is_inquiry and estate_rag.answer:
                safe_ans = html.escape(estate_rag.answer.strip())
                formatted_ans = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", safe_ans)
                formatted_ans = re.sub(r"### (.*?)\n", r"\n<b>\1</b>\n", formatted_ans)
                reply_body += f"\n📖 <b>Estate Policy Intelligence:</b>\n{formatted_ans}\n"

            infras = [i for i in estate_rag.affected_infrastructure if "Unit #" not in i]
            if infras:
                safe_infras = html.escape(", ".join(infras[:3]))
                reply_body += f"\n🔧 <b>Connected Infrastructure:</b> {safe_infras}\n"
            if estate_rag.by_laws_cited:
                safe_bylaws = html.escape(", ".join(estate_rag.by_laws_cited[:2]))
                reply_body += f"⚖️ <b>Governing By-Law:</b> {safe_bylaws}\n"

        if decision.is_emergency:
            reply_body += (
                "\n📲 <i>Dispatched via on-call contractor SMS & guardhouse alert.</i>\n"
                "⚠️ <i>[PROTOTYPE DISPATCH: For actual life-threatening emergencies, "
                "call 995/999 immediately.]</i>"
            )

        # Inline Action Buttons with safe byte length
        safe_cb_contact = decision.entity_contact[:45].strip()
        safe_cb_wo = work_order[:45].strip()
        keyboard = [
            [
                InlineKeyboardButton(
                    "🚰 Emergency Water Shutoff Guide",
                    callback_data="guide_water",
                ),
                InlineKeyboardButton(
                    "📞 Call Assigned Tech",
                    callback_data=f"call_{safe_cb_contact}",
                ),
            ],
            [
                InlineKeyboardButton(
                    "📋 View Live Work Order",
                    callback_data=f"view_{safe_cb_wo}",
                )
            ],
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        await message.reply_text(reply_body, parse_mode=ParseMode.HTML, reply_markup=reply_markup)

    except Exception as exc:
        logger.error(f"Error handling message: {exc}", exc_info=True)
        await message.reply_text(
            "⚠️ Error processing request. Please contact guardhouse directly at +65 6444-2222."
        )


async def handle_callback_query(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if not query:
        return
    await query.answer()

    data = query.data or ""
    if data == "guide_water":
        vault = get_estate_vault()
        rag_res = vault.query("emergency water isolation shutoff valve leak protocol", hops=1)
        infra_mention = ""
        if rag_res.affected_infrastructure:
            infra_list = ", ".join(f"<code>{i}</code>" for i in rag_res.affected_infrastructure[:3])
            infra_mention = f"• <b>Estate Infrastructure:</b> {infra_list}\n"
        bylaw_mention = ""
        if rag_res.by_laws_cited:
            bylaw_mention = f"• <b>Governing By-Law:</b> {', '.join(rag_res.by_laws_cited[:2])}\n"

        await query.message.reply_text(
            "🚰 <b>Estate Water Isolation & Emergency Guide</b>\n\n"
            "1. Locate your unit's main isolation valve in the service riser or beneath the sink.\n"
            "2. Turn the circular valve handle clockwise until tightly closed.\n"
            "3. Relieve residual water pressure by opening a cold water tap.\n"
            "4. Keep main door unlocked for dispatched emergency technicians.\n\n"
            f"{infra_mention}"
            f"{bylaw_mention}"
            "📞 <b>Emergency Hotline:</b> <code>+65 6789-0123</code> (Apex 24/7 Rapid Plumbing)",
            parse_mode=ParseMode.HTML,
        )
    elif data.startswith("call_"):
        number = html.escape(data.replace("call_", ""))
        await query.message.reply_text(
            f"📞 Contact number for assigned technician: <code>{number}</code>",
            parse_mode=ParseMode.HTML,
        )
    elif data.startswith("view_"):
        wo_id = data.replace("view_", "").strip()
        ticket = get_ticket(wo_id)
        if not ticket:
            await query.message.reply_text(
                f"❌ <b>Work Order <code>{html.escape(wo_id)}</code> not found in database.</b>",
                parse_mode=ParseMode.HTML,
            )
            return

        clearance = (
            "Pre-cleared at Guardhouse ✅"
            if ticket.get("priority") == "emergency"
            else "Standard Entry"
        )
        cat_name = ticket["category"].title()
        prio_name = ticket["priority"].upper()
        team_name = ticket.get("assigned_entity") or "Pending Assignment"
        phone_contact = ticket.get("entity_contact") or "N/A"
        await query.message.reply_text(
            f"📋 <b>Work Order Status: {html.escape(wo_id)}</b>\n\n"
            f"• <b>Live Status:</b> {html.escape(ticket['status'])}\n"
            f"• <b>Assigned Team:</b> {html.escape(team_name)}\n"
            f"• <b>Contact:</b> <code>{html.escape(phone_contact)}</code>\n"
            f"• <b>Guardhouse Clearance:</b> {clearance}\n"
            f"• <b>Category:</b> {html.escape(cat_name)} | "
            f"<b>Priority:</b> {html.escape(prio_name)}",
            parse_mode=ParseMode.HTML,
        )
    elif data.startswith("rate_"):
        parts = data.split("_")
        if len(parts) >= 3 and query.message:
            t_id = parts[1]
            try:
                stars = int(parts[2])
            except ValueError:
                stars = 5
            record_resident_feedback(t_id, stars)
            star_emojis = "⭐" * stars
            await query.message.reply_text(
                f"🌟 <b>Thank you for your rating!</b>\n\n"
                f"You rated Work Order <code>{html.escape(t_id)}</code>: "
                f"{star_emojis} ({stars}/5).\n"
                "Your feedback helps our condominium management improve service quality.",
                parse_mode=ParseMode.HTML,
            )


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    # Ensure database tables exist
    init_db()

    if not TELEGRAM_BOT_TOKEN:
        print("\n" + "=" * 70)
        print(" [!] Missing TELEGRAM_BOT_TOKEN in .env file.")
        print(" Please add TELEGRAM_BOT_TOKEN=your_token_here to .env and re-run.")
        print("=" * 70 + "\n")
        return

    print("=" * 70)
    print(" [*] Starting Condo Agent: Autonomous Emergency Coordinator Bot...")
    print("=" * 70)

    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("ask", ask_command))
    app.add_handler(CommandHandler("status", status_command))
    app.add_handler(CommandHandler("mytickets", mytickets_command))
    app.add_handler(
        MessageHandler(filters.TEXT | filters.PHOTO & ~filters.COMMAND, handle_resident_message)
    )
    app.add_handler(CallbackQueryHandler(handle_callback_query))

    print(" [*] Bot is online! Listening for resident messages on Telegram...")
    app.run_polling()


if __name__ == "__main__":
    main()
