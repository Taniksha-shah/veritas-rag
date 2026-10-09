"""
Audit and Compliance Logging Module for Veritas RAG Engine.
Maintains auditable traces of every verification decision and iteration.
"""

from veritas.audit.logger import AuditLogger, AuditRecord

__all__ = ["AuditLogger", "AuditRecord"]
