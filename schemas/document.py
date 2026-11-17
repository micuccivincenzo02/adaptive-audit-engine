from src.models import ContractClause, DocumentFact, DocumentMeta, LineItem

# Backward-compatible aliases — DocumentFact handles all document types
Contract = DocumentFact
Invoice = DocumentFact
DDT = DocumentFact

__all__ = [
    "Contract",
    "ContractClause",
    "DDT",
    "DocumentFact",
    "DocumentMeta",
    "Invoice",
    "LineItem",
]
