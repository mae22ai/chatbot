from django.core.management.base import BaseCommand
from chatbot_app.services import process_chatbot_request
import json

class Command(BaseCommand):
    help = 'Test the chatbot analyzer pipeline with various flags'

    def add_arguments(self, parser):
        parser.add_argument('text', type=str, help='Input sentence to analyze')
        parser.add_argument('--bareun', action='store_true', help='Enable Bareun analyzer')
        parser.add_argument('--llm', action='store_true', help='Enable LLM analyzer')
        parser.add_argument('--heuristics', action='store_true', help='Enable Heuristics')
        parser.add_argument('--all', action='store_true', help='Enable ALL components')

    def handle(self, *args, **options):
        text = options['text']
        
        # Determine flags based on arguments
        if options['all']:
            use_bareun = True
            use_llm = True
            use_heuristics = True
        else:
            use_bareun = options['bareun']
            use_llm = options['llm']
            use_heuristics = options['heuristics']

        self.stdout.write(self.style.SUCCESS(f"Analyzing: '{text}'"))
        self.stdout.write(f"Flags: Bareun={use_bareun}, LLM={use_llm}, Heuristics={use_heuristics}")
        self.stdout.write("-" * 40)

        try:
            result = process_chatbot_request(
                text=text,
                use_bareun=use_bareun,
                use_heuristics=use_heuristics,
                use_llm=use_llm
            )

            if result['ok']:
                if use_llm:
                    self.stdout.write(self.style.SUCCESS("Analysis Result (Markdown):"))
                    self.stdout.write(result['markdown'])
                else:
                    self.stdout.write(self.style.WARNING("LLM Skipped. Intermediate Data:"))
                    debug_info = result.get('debug_info', {})
                    self.stdout.write(f"Pos Line: {debug_info.get('pos_line')}")
                    self.stdout.write(f"Heuristics: \n{debug_info.get('heuristic_info')}")
            else:
                self.stdout.write(self.style.ERROR(f"Error: {result.get('error')}"))

        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Command failed: {e}"))
