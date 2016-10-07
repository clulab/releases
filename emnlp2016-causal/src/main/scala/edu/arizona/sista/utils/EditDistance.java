package edu.arizona.sista.utils;

public class EditDistance {
	public static void main(String[] args) {
		System.out.printf("distance between \"%s\" and \"%s\" is %d\n", args[0], args[1], levenshtein(args[0], args[1]));
	}
	
	public static int levenshtein(String s1, String s2) {
		int [][] m = new int[s1.length() + 1][s2.length() + 1];
		m[0][0] = 0;
		for(int i = 1; i <= s1.length(); i ++) m[i][0] = i + 1;
		for(int j = 1; j <= s2.length(); j ++) m[0][j] = j + 1;
		
		for(int i = 1; i <= s1.length(); i ++){
			for(int j = 1; j <= s2.length(); j ++){
				m[i][j] = min(
						m[i - 1][j - 1] + (s1.charAt(i - 1) == s2.charAt(j - 1) ? 0 : 1),
						m[i - 1][j] + 1,
						m[i][j - 1] + 1);
			}
		}
		
		return m[s1.length()][s2.length()];
	}
	
	private static int min(int i1, int i2, int i3){
		return Math.min(i1, Math.min(i2, i3));
	}
}
