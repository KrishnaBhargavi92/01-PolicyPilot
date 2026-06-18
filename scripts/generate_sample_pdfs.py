from pathlib import Path
from textwrap import wrap


ROOT_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT_DIR / "data" / "docs"

POLICIES = {
    "employment-eligibility-verification-policy.pdf": {
        "title": "Employment Eligibility Verification Policy",
        "sections": [
            (
                "Policy Statement",
                "The organization verifies employment eligibility for all employees "
                "and workers in accordance with applicable law. New workers must "
                "complete required eligibility forms and provide acceptable identity "
                "and work authorization documents within the required timeframe.",
            ),
            (
                "Responsibilities",
                "Workers are responsible for completing their portion of the "
                "verification form by the first day of work. Authorized reviewers "
                "are responsible for reviewing original documents and completing "
                "the employer portion by the third business day after work begins.",
            ),
            (
                "Compliance",
                "A worker may not continue working beyond the permitted verification "
                "period if required documentation has not been completed. Questions "
                "should be directed to the human resources team.",
            ),
        ],
    },
    "hr-record-retention-policy.pdf": {
        "title": "HR Record Retention Policy",
        "sections": [
            (
                "Policy Statement",
                "The organization retains human resources records for business, "
                "operational, legal, and compliance purposes. Records are maintained "
                "securely and disposed of when retention requirements are satisfied.",
            ),
            (
                "Retention",
                "Employee records, payroll records, benefit records, recruiting "
                "records, and workplace investigation records are retained according "
                "to the applicable retention schedule.",
            ),
            (
                "Access",
                "Access to personnel records is limited to authorized personnel with "
                "a legitimate business need. Confidential records must not be shared "
                "outside approved processes.",
            ),
        ],
    },
    "lactation-support-policy.pdf": {
        "title": "Lactation Support Policy",
        "sections": [
            (
                "Policy Statement",
                "The organization provides reasonable lactation support for eligible "
                "workers. Support may include break time and access to a private, "
                "non-bathroom space for expressing milk.",
            ),
            (
                "Process",
                "Workers may request lactation support through human resources or "
                "their manager. Requests are handled promptly and confidentially.",
            ),
            (
                "Non-Retaliation",
                "The organization prohibits retaliation against workers who request "
                "or use lactation support.",
            ),
        ],
    },
    "paid-holidays-policy.pdf": {
        "title": "Paid Holidays Policy",
        "sections": [
            (
                "Policy Statement",
                "The organization observes designated paid holidays each calendar "
                "year. Eligibility for paid holidays depends on worker classification, "
                "schedule, and applicable policy terms.",
            ),
            (
                "Scheduling",
                "If business needs require work on a designated holiday, managers "
                "will coordinate schedules and any applicable alternate time off or "
                "premium pay in accordance with policy.",
            ),
            (
                "Administration",
                "Human resources publishes the annual holiday schedule and answers "
                "questions about eligibility and pay treatment.",
            ),
        ],
    },
    "pregnancy-support-policy.pdf": {
        "title": "Pregnancy Support Policy",
        "sections": [
            (
                "Policy Statement",
                "The organization provides reasonable support for pregnancy, "
                "childbirth, and related medical conditions. Requests are evaluated "
                "through an interactive process.",
            ),
            (
                "Examples",
                "Support may include schedule adjustments, modified duties, seating, "
                "access to water, additional breaks, or other reasonable workplace "
                "changes based on individual needs.",
            ),
            (
                "Confidentiality",
                "Medical information is handled confidentially and shared only with "
                "personnel who need the information to administer support.",
            ),
        ],
    },
    "staff-background-check-policy.pdf": {
        "title": "Staff Background Check Policy",
        "sections": [
            (
                "Policy Statement",
                "The organization may conduct background checks for designated roles "
                "to support workplace safety, compliance, and responsible hiring.",
            ),
            (
                "Scope",
                "Background checks may apply to new hires, current workers changing "
                "roles, contractors, or other individuals based on job duties and "
                "legal requirements.",
            ),
            (
                "Review",
                "Results are reviewed consistently, confidentially, and in accordance "
                "with applicable law. Individuals are provided required notices and "
                "opportunities to respond where applicable.",
            ),
        ],
    },
    "staff-loa-policy.pdf": {
        "title": "Staff Leave of Absence Policy",
        "sections": [
            (
                "Policy Statement",
                "The organization provides eligible workers with leaves of absence "
                "for qualifying personal, medical, family, military, or other "
                "approved reasons.",
            ),
            (
                "Request Process",
                "Workers should request leave as soon as the need is known. Human "
                "resources will review eligibility, documentation requirements, pay "
                "status, benefit impact, and return-to-work expectations.",
            ),
            (
                "Return To Work",
                "Workers are expected to communicate about their return date and "
                "provide required documentation before returning when applicable.",
            ),
        ],
    },
}


def escape_pdf_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def render_page(title: str, lines: list[str], page_number: int) -> bytes:
    commands = ["BT", "/F1 16 Tf", "72 740 Td", f"({escape_pdf_text(title)}) Tj"]
    commands.extend(["/F1 10 Tf", "0 -28 Td"])

    for line in lines:
        commands.append(f"({escape_pdf_text(line)}) Tj")
        commands.append("0 -15 Td")

    commands.extend(["0 -20 Td", f"(Page {page_number}) Tj", "ET"])
    stream = "\n".join(commands).encode("utf-8")
    return (
        b"<< /Length "
        + str(len(stream)).encode("ascii")
        + b" >>\nstream\n"
        + stream
        + b"\nendstream"
    )


def write_pdf(path: Path, title: str, body_lines: list[str]) -> None:
    lines_per_page = 40
    pages = [
        body_lines[index : index + lines_per_page]
        for index in range(0, len(body_lines), lines_per_page)
    ]

    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        None,
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]

    page_object_numbers = []
    for page_number, page_lines in enumerate(pages, start=1):
        content_object_number = len(objects) + 2
        page_object_numbers.append(len(objects) + 1)
        objects.append(
            (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                f"/Resources << /Font << /F1 3 0 R >> >> "
                f"/Contents {content_object_number} 0 R >>"
            ).encode("ascii")
        )
        objects.append(render_page(title, page_lines, page_number))

    kids = " ".join(f"{number} 0 R" for number in page_object_numbers)
    objects[1] = f"<< /Type /Pages /Kids [{kids}] /Count {len(page_object_numbers)} >>".encode(
        "ascii"
    )

    output = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for object_number, content in enumerate(objects, start=1):
        offsets.append(len(output))
        output.extend(f"{object_number} 0 obj\n".encode("ascii"))
        output.extend(content)
        output.extend(b"\nendobj\n")

    xref_offset = len(output)
    output.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    output.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode("ascii"))

    output.extend(
        (
            f"trailer << /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode("ascii")
    )
    path.write_bytes(output)


def policy_lines(title: str, sections: list[tuple[str, str]]) -> list[str]:
    lines = [
        "Sample Policy Document",
        "This document is anonymized sample content for demonstration purposes.",
        "It does not reference a real company, institution, client, person, or location.",
        "",
    ]

    for heading, text in sections:
        lines.append(heading)
        lines.extend(wrap(text, width=88))
        lines.append("")

    lines.extend(
        [
            "Administration",
            "Questions about this sample policy may be directed to the human resources team.",
            "",
            f"Document Title: {title}",
        ]
    )
    return lines


def main() -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    for filename, policy in POLICIES.items():
        path = DOCS_DIR / filename
        write_pdf(path, policy["title"], policy_lines(policy["title"], policy["sections"]))
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
